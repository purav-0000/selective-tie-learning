"""Synthetic hotel preference task with controlled nuisance correlations.

The nuisance level is the empirical CDF of latent hotel utility. Reversing
that level changes its coupling to utility while preserving nuisance
marginals across the training, clean, and adversarial environments.
"""


import copy
import numpy as np
import random

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

class PreferenceLabel(Enum):
    """Which generated hotel has greater latent utility."""

    A_BETTER = "A"
    B_BETTER = "B"

# Seven fields rendered in every hotel profile but excluded from utility.
SPURIOUS_FEATURES = (
    "street_number", "floor_number", "building_age", "renovation_year",
    "hotel_chain_tier", "lobby_size_sqft", "employee_count")


def _tier_at_level(level: float) -> str:
    if level < 0.33:
        return "Budget"
    if level < 0.67:
        return "Standard"
    return "Premium"


# A level of one gives the most desirable value for every nuisance field.
_SPURIOUS_LEVEL_MAPS = {
    "street_number": lambda lv: int(100 + lv * (9999 - 100)),
    "floor_number": lambda lv: int(1 + lv * (20 - 1)),
    "building_age": lambda lv: int(50 - int(lv * (50 - 1))),
    "renovation_year": lambda lv: int(2000 + int(lv * 24)),
    "hotel_chain_tier": _tier_at_level,
    "lobby_size_sqft": lambda lv: int(500 + int(lv * (5000 - 500))),
    "employee_count": lambda lv: int(10 + int(lv * (200 - 10))),
}


def spurious_feature_dict(hotel: "Hotel", fields=SPURIOUS_FEATURES) -> Dict:
    """The spurious block of one hotel, for dataset diagnostics."""
    return {f: getattr(hotel, f) for f in fields}


def randomize_spurious_features(hotel: "Hotel", rng=random,
                                fields=SPURIOUS_FEATURES) -> "Hotel":
    """Assign every spurious feature an independent uniform level.

    Uses the same per-feature level maps as the correlated assignment, so
    randomized and correlated hotels have exactly matching marginals.
    """
    for field in fields:
        setattr(hotel, field, _SPURIOUS_LEVEL_MAPS[field](rng.random()))
    return hotel


def copy_spurious_features(target: "Hotel", donor: "Hotel",
                           fields=SPURIOUS_FEATURES) -> "Hotel":
    """Replace only the designated nuisance block with a donor's block."""
    for field in fields:
        setattr(target, field, copy.deepcopy(getattr(donor, field)))
    return target


@dataclass
class Hotel:
    """Complete hotel profile and its persistent latent-utility noise."""
    name: str
    street_number: int
    street_name: str
    distance_to_convention_center: float  # in miles
    price_per_night: int
    star_rating: float
    has_pool: bool
    has_gym: bool
    has_breakfast: bool
    has_parking: bool
    room_size_sqft: int
    review_score: float                   # 1-10
    floor_number: int
    building_age: int                     # years since construction
    renovation_year: int                  # last renovation year
    hotel_chain_tier: str                 # "Premium", "Standard", "Budget"
    lobby_size_sqft: int                  # lobby size in square feet
    employee_count: int                   # number of employees
    utility_noise: float = 0.0             # persistent latent utility component

    def to_description(self) -> str:
        """Render every causal and nuisance field as natural language."""
        amenities = []
        if self.has_pool:
            amenities.append("pool")
        if self.has_gym:
            amenities.append("gym")
        if self.has_breakfast:
            amenities.append("complimentary breakfast")
        if self.has_parking:
            amenities.append("free parking")

        amenities_str = ", ".join(amenities) if amenities else "no special amenities"

        return (
            f"{self.name} is prominently located at {self.street_number} {self.street_name}. "
            f"This {self.hotel_chain_tier} hotel, built {self.building_age} years ago and renovated in {self.renovation_year}, "
            f"features a {self.lobby_size_sqft} square foot lobby and is staffed by {self.employee_count} employees. "
            f"The property at {self.street_number} {self.street_name} is {self.distance_to_convention_center:.1f} miles from the convention center. "
            f"It costs ${self.price_per_night} per night and has a {self.star_rating} star rating. "
            f"The hotel features {amenities_str}. "
            f"Rooms are {self.room_size_sqft} square feet on floor {self.floor_number}. "
            f"Guests staying on floor {self.floor_number} at this {self.hotel_chain_tier} property with {self.employee_count} staff members "
            f"have given it a review score of {self.review_score:.1f}/10."
        )


# The spurious level of a hotel is the empirical CDF of its true utility,
# estimated once by Monte Carlo with an isolated RNG so the calibration is
# identical for every generator instance and does not disturb the seeded
# data-generation stream. Levels are therefore Uniform(0,1) under the
# training distribution, which makes the per-feature spurious marginals
# identical across the normal / suppressed / adversarial modes.
_UTILITY_CDF_SEED = 20260712
_UTILITY_CDF_SAMPLES = 100_000
_UTILITY_CDF_CACHE: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}


class HotelDatasetGenerator:
    """Generate strict hotel pairs under a selected correlation environment."""

    def __init__(self, spurious_correlation_strength: float = 0.8,
                 utility_noise_std: float = 2.5,
                 strict_min_gap: float = 5.0,
                 spurious_seed: int | None = None,
                 spurious_features=SPURIOUS_FEATURES):
        """Configure correlation strength, utility noise, and strictness gap."""
        self.spurious_correlation_strength = spurious_correlation_strength
        self.utility_noise_std = utility_noise_std
        self.strict_min_gap = strict_min_gap
        self.spurious_features = tuple(spurious_features)
        if (not self.spurious_features or
                len(set(self.spurious_features)) != len(self.spurious_features) or
                any(field not in _SPURIOUS_LEVEL_MAPS
                    for field in self.spurious_features)):
            raise ValueError("spurious_features must be a nonempty unique subset")
        self.spurious_rng = (random.Random(spurious_seed)
                             if spurious_seed is not None else random)
        self.street_names = [
            "Main St", "Oak Ave", "Park Blvd", "Center Dr", "Market St",
            "Union Sq", "Broadway", "First Ave", "Second Ave", "Third Ave"
        ]
        self.hotel_chains = [
            "Hilton", "Marriott", "Hyatt", "Holiday Inn", "Best Western",
            "Sheraton", "Radisson", "Comfort Inn", "Hampton Inn", "DoubleTree"
        ]

    def generate_random_hotel(self, rng=random) -> Hotel:
        """Sample a complete hotel before applying nuisance correlation."""
        return Hotel(
            name=f"{rng.choice(self.hotel_chains)} {rng.choice(['Downtown', 'Central', 'Plaza', 'Suites', 'Express'])}",
            street_number=rng.randint(100, 9999),
            street_name=rng.choice(self.street_names),
            distance_to_convention_center=round(rng.uniform(0.1, 10.0), 1),
            price_per_night=rng.randint(50, 500),
            star_rating=rng.choice([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
            has_pool=rng.random() > 0.5,
            has_gym=rng.random() > 0.5,
            has_breakfast=rng.random() > 0.5,
            has_parking=rng.random() > 0.6,
            room_size_sqft=rng.randint(200, 600),
            review_score=round(rng.uniform(5.0, 10.0), 1),
            floor_number=rng.randint(1, 20),
            building_age=rng.randint(1, 50),
            renovation_year=rng.randint(2000, 2024),
            hotel_chain_tier=rng.choice(["Premium", "Standard", "Budget"]),
            lobby_size_sqft=rng.randint(500, 5000),
            employee_count=rng.randint(10, 200),
            utility_noise=rng.gauss(0.0, self.utility_noise_std),
        )

    def _utility_cdf_samples(self, context: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """CDF of hotels that survive the strict-pair acceptance rule.

        Conditioning on ``|u_a-u_b| >= strict_min_gap`` changes the hotel
        utility distribution, so unconditional calibration is inappropriate.
        It changes the marginal utility distribution of hotels that appear in
        the generated train and evaluation sets.
        The force-balanced label rule can change it again because the sampler
        resamples B while holding A fixed.  We reproduce both acceptance rules
        with an isolated RNG and collect an equal number of A-better and
        B-better pairs.  Consequently the probability-integral-transform is
        uniform for the *observed strict population*, which is the population
        whose coupling is flipped in Q_adv.
        """
        key = (str(context.get("requirements", "")).lower(),
               round(self.utility_noise_std, 12),
               round(self.strict_min_gap, 12))
        cached = _UTILITY_CDF_CACHE.get(key)
        if cached is None:
            rng = random.Random(_UTILITY_CDF_SEED)
            target_pairs_per_label = max(_UTILITY_CDF_SAMPLES // 4, 1)
            counts = {PreferenceLabel.A_BETTER: 0,
                      PreferenceLabel.B_BETTER: 0}
            utilities = []
            while min(counts.values()) < target_pairs_per_label:
                hotel_a = self.generate_random_hotel(rng)
                hotel_b = self.generate_random_hotel(rng)
                util_a = self.calculate_true_utility(hotel_a, context)
                util_b = self.calculate_true_utility(hotel_b, context)
                attempts = 0
                while (abs(util_a - util_b) < self.strict_min_gap
                       and attempts < 50):
                    hotel_b = self.generate_random_hotel(rng)
                    util_b = self.calculate_true_utility(hotel_b, context)
                    attempts += 1
                if abs(util_a - util_b) < self.strict_min_gap:
                    continue
                label = (PreferenceLabel.A_BETTER if util_a > util_b
                         else PreferenceLabel.B_BETTER)
                if counts[label] >= target_pairs_per_label:
                    continue
                counts[label] += 1
                utilities.extend((util_a, util_b))
            samples = np.sort(np.asarray(utilities, dtype=float))
            ranks = (np.arange(len(samples)) + 0.5) / len(samples)
            cached = (samples, ranks)
            _UTILITY_CDF_CACHE[key] = cached
        return cached

    def utility_level(self, utility: float, context: Dict) -> float:
        """Empirical-CDF spurious level of ``utility`` in [0, 1].

        ``np.interp`` clamps to the sampled utility range, so extreme
        utilities map to levels near 0 or 1 (never outside), and the level of
        a uniformly drawn strict-population hotel is Uniform(0, 1) by the
        probability integral transform. The calibration population is keyed on
        ``(context, utility_noise_std, strict_min_gap)`` -- any caller that
        needs levels consistent with a strict/test file must construct its
        generator with the *same* three values.
        """
        samples, ranks = self._utility_cdf_samples(context)
        return float(np.interp(utility, samples, ranks))

    def calculate_true_utility(self, hotel: Hotel, context: Dict) -> float:
        """Compute utility from causal fields and persistent sampled noise.

        Price, distance, star rating, and requested amenities affect utility.
        Review, room size, and the seven nuisance fields do not.
        """
        utility = 0.0

        # Distance: closer is better (max 10 points).
        distance_score = max(0, 10 - hotel.distance_to_convention_center)
        utility += distance_score

        # Price: lower is better (max 10 points).
        price_score = max(0, 10 - (hotel.price_per_night / 50))
        utility += price_score

        # Star rating: the task requires at least three stars.
        if hotel.star_rating >= 3.0:
            utility += 5.0
        else:
            utility -= 10.0

        requirements = str(context.get("requirements", "")).lower()
        for keyword, present in (
            ("pool", hotel.has_pool),
            ("gym", hotel.has_gym),
            ("breakfast", hotel.has_breakfast),
            ("parking", hotel.has_parking),
        ):
            if keyword in requirements:
                utility += 2.5 if present else -2.5

        return utility + hotel.utility_noise

    def add_spurious_correlations(
        self,
        hotel: Hotel,
        utility: float,
        context: Dict,
        correlation_mode: str = "normal"
        ) -> Tuple[Hotel, Dict]:
        """Couple a hotel's nuisance block to its empirical utility rank."""
        # Spurious level = empirical CDF of utility, so the level is
        # Uniform(0,1) under the training distribution and the adversarial
        # flip level -> 1-level preserves every spurious marginal.
        utility_level = self.utility_level(utility, context)
        tracking = {
            'correlation_mode': correlation_mode,
            'utility': utility,
            'utility_level': utility_level,
            'correlation_applied': False,
            'random_roll': None,
            'threshold': self.spurious_correlation_strength
        }

        rng = self.spurious_rng
        random_roll = rng.random()
        tracking['random_roll'] = random_roll

        if random_roll >= self.spurious_correlation_strength:
            # With probability (1-ρ), assign randomly
            tracking['correlation_applied'] = False
            tracking['correlation_direction'] = 'none'
            randomize_spurious_features(hotel, rng, self.spurious_features)
            tracking['assignment_type'] = 'random_noise'
            tracking['spurious_features'] = spurious_feature_dict(hotel)
            tracking['active_spurious_features'] = list(self.spurious_features)
            return hotel, tracking

        tracking['correlation_applied'] = True

        if correlation_mode == "clean":
            # Preserve the joint nuisance marginal while breaking its link to
            # utility.
            independent_level = rng.random()
            apply_spurious_features_at_level(
                hotel, independent_level, "normal", self.spurious_features)
            tracking['correlation_direction'] = 'none'
            tracking['correlation_applied'] = False
            tracking['assignment_type'] = 'independent_shared_level'
            tracking['independent_level'] = independent_level
        elif correlation_mode == "adversarial":
            tracking['correlation_direction'] = 'inverted'
            tracking['assignment_type'] = 'adversarial_continuous'
            apply_spurious_features_at_level(
                hotel, utility_level, correlation_mode, self.spurious_features)
        elif correlation_mode == "normal":
            tracking['correlation_direction'] = 'normal'
            tracking['assignment_type'] = 'normal_continuous'
            apply_spurious_features_at_level(
                hotel, utility_level, correlation_mode, self.spurious_features)
        else:
            raise ValueError(f"unknown correlation_mode={correlation_mode!r}")
        tracking['spurious_features'] = spurious_feature_dict(hotel)
        tracking['active_spurious_features'] = list(self.spurious_features)

        return hotel, tracking

    def generate_hotel_pair(
        self,
        context: Dict,
        correlation_mode: str = "normal"
        ) -> Tuple[Hotel, Hotel, PreferenceLabel, Dict]:
        """Sample and orient a pair whose latent-utility gap is large enough."""
        hotel_a = self.generate_random_hotel()
        hotel_b = self.generate_random_hotel()
        util_a = self.calculate_true_utility(hotel_a, context)
        util_b = self.calculate_true_utility(hotel_b, context)

        attempts = 0
        while abs(util_a - util_b) < self.strict_min_gap and attempts < 50:
            hotel_b = self.generate_random_hotel()
            util_b = self.calculate_true_utility(hotel_b, context)
            attempts += 1
        if abs(util_a - util_b) < self.strict_min_gap:
            raise RuntimeError(
                f"failed to sample strict pair with gap >= {self.strict_min_gap}")
        label = PreferenceLabel.A_BETTER if util_a > util_b else PreferenceLabel.B_BETTER

        # Labels are fixed before nuisance fields are assigned.
        hotel_a, tracking_a = self.add_spurious_correlations(
            hotel_a, util_a, context, correlation_mode)
        hotel_b, tracking_b = self.add_spurious_correlations(
            hotel_b, util_b, context, correlation_mode)

        pair_tracking = {
            'hotel_a': tracking_a,
            'hotel_b': tracking_b,
            'both_correlated': tracking_a['correlation_applied'] and tracking_b['correlation_applied'],
            'neither_correlated': not tracking_a['correlation_applied'] and not tracking_b['correlation_applied'],
            'mixed_correlation': (tracking_a['correlation_applied'] != tracking_b['correlation_applied'])
        }

        return hotel_a, hotel_b, label, pair_tracking

def apply_spurious_features_at_level(
    hotel: 'Hotel',
    utility_level: float,
    correlation_mode: str = "normal",
    fields=SPURIOUS_FEATURES,
) -> 'Hotel':
    """Set every nuisance field from one desirability level in ``[0, 1]``."""
    if correlation_mode == "adversarial":
        utility_level = 1.0 - utility_level

    # One shared level preserves dependence among the seven nuisance fields.
    for field in fields:
        setattr(hotel, field, _SPURIOUS_LEVEL_MAPS[field](utility_level))

    return hotel


def format_dpo_example(
    context_prompt: str,
    option_a_text: str,
    option_b_text: str,
    util_a: float,
    util_b: float,
    label: PreferenceLabel,
    ) -> Dict:
    """Render a pairwise example after randomizing the displayed positions."""

    response_one = "Option ONE is the better choice."
    response_two = "Option TWO is the better choice."

    metadata = {}

    if random.random() > 0.5:
        prompt_text_one = option_a_text
        prompt_text_two = option_b_text
        metadata['option_one_utility'] = util_a
        metadata['option_two_utility'] = util_b

        if label == PreferenceLabel.A_BETTER:
            chosen, rejected = response_one, response_two
            metadata['true_label'] = "A"
        else:
            chosen, rejected = response_two, response_one
            metadata['true_label'] = "B"
    else:
        prompt_text_one = option_b_text
        prompt_text_two = option_a_text
        metadata['option_one_utility'] = util_b
        metadata['option_two_utility'] = util_a

        if label == PreferenceLabel.A_BETTER:
            chosen, rejected = response_two, response_one
            metadata['true_label'] = "B"
        else:
            chosen, rejected = response_one, response_two
            metadata['true_label'] = "A"

    full_prompt = (
        f"{context_prompt}\n\n"
        "Here are two options:\n\n"
        "--- Option ONE ---\n"
        f"{prompt_text_one}\n\n"
        "--- Option TWO ---\n"
        f"{prompt_text_two}\n\n"
        "--- Task ---\n"
        "Which of these two options is the better choice for the user?\n\n"
        "--- Answer ---\n"
    )

    return {
        "prompt": full_prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": metadata,
    }
