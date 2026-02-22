"""
Content-based anime recommender system using manual algorithm.
Leverages Jaccard distance, metadata weighting, and user ratings.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class AnimeMetadata:
    """Structured metadata for an anime."""
    mal_id: str
    title: str
    title_english: str
    my_score: float
    mal_score: float
    genres: Set[str] = field(default_factory=set)
    episodes: int = 0
    studios: Set[str] = field(default_factory=set)
    synopsis: str = ""
    my_status: str = ""
    type_: str = ""  # TV, Movie, etc.
    aired_from: str = ""  # Release date in YYYY-MM-DD format

    @classmethod
    def from_json(cls, anime_dict: Dict) -> "AnimeMetadata":
        """Create AnimeMetadata from JSON dictionary."""
        genres = {g["name"] for g in anime_dict.get("genres", [])}
        studios = {s["name"] for s in anime_dict.get("studios", [])}
        
        return cls(
            mal_id=anime_dict.get("mal_id", ""),
            title=anime_dict.get("title", ""),
            title_english=anime_dict.get("title_english", ""),
            my_score=float(anime_dict.get("my_score", 0)),
            mal_score=float(anime_dict.get("mal_score", 0)),
            genres=genres,
            episodes=int(anime_dict.get("episodes", 0)),
            studios=studios,
            synopsis=anime_dict.get("synopsis", ""),
            my_status=anime_dict.get("my_status", ""),
            type_=anime_dict.get("type", ""),
            aired_from=anime_dict.get("aired_from", ""),
        )


class JaccardSimilarity:
    """Computes Jaccard similarity between sets."""
    
    @staticmethod
    def similarity(set_a: Set[str], set_b: Set[str]) -> float:
        """
        Compute Jaccard similarity between two sets.
        J(A,B) = |A ∩ B| / |A ∪ B|
        Returns 0 if both sets are empty.
        """
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return intersection / union if union > 0 else 0.0


class ContentBasedRecommender:
    """Content-based recommender system using manual algorithm."""
    
    def __init__(
        self,
        weight_genre: float = 0.35,
        weight_user_score: float = 0.25,
        weight_mal_score: float = 0.15,
        weight_studio: float = 0.05,
        weight_type: float = 0.03,
        weight_episodes: float = 0.02,
        weight_recency: float = 0.15,
        min_user_rating: float = 6.0,
        normalize_scores: bool = True,
        recency_decay_years: float = 5.0,
    ):
        """
        Initialize recommender with configurable weights.
        
        Args:
            weight_genre: Weight for genre similarity (Jaccard)
            weight_user_score: Weight for user rating alignment
            weight_mal_score: Weight for MAL community score
            weight_studio: Weight for studio similarity
            weight_type: Weight for anime type matching
            weight_episodes: Weight for episode count similarity
            weight_recency: Weight for release date recency
            min_user_rating: Minimum user rating to consider for reference (completed anime)
            normalize_scores: Whether to normalize scores to [0, 1]
            recency_decay_years: Years for recency decay (anime older than this get score close to 0)
        """
        # Normalize weights to sum to 1
        total_weight = (
            weight_genre + weight_user_score + weight_mal_score + 
            weight_studio + weight_type + weight_episodes + weight_recency
        )
        
        self.weight_genre = weight_genre / total_weight
        self.weight_user_score = weight_user_score / total_weight
        self.weight_mal_score = weight_mal_score / total_weight
        self.weight_studio = weight_studio / total_weight
        self.weight_type = weight_type / total_weight
        self.weight_episodes = weight_episodes / total_weight
        self.weight_recency = weight_recency / total_weight
        
        self.min_user_rating = min_user_rating
        self.normalize_scores = normalize_scores
        self.recency_decay_years = recency_decay_years
        
        self.user_profile = None
        self.completed_anime = []
        self.reference_date = None  # Current date for recency calculations
        
    def build_profile(self, anime_list: List[AnimeMetadata]) -> None:
        """
        Build user profile from completed anime with sufficient ratings.
        
        Args:
            anime_list: List of AnimeMetadata objects
        """
        from datetime import datetime
        
        self.completed_anime = [
            anime for anime in anime_list
            if anime.my_status == "Completed" and anime.my_score >= self.min_user_rating
        ]
        
        # Set reference date to today for recency calculations
        self.reference_date = datetime.now()
        
        if not self.completed_anime:
            raise ValueError(
                f"No completed anime with rating >= {self.min_user_rating} found. "
                "Lower min_user_rating to build a profile."
            )
        
        # Aggregate user preferences
        all_genres = set()
        all_studios = set()
        avg_episodes = np.mean([a.episodes for a in self.completed_anime])
        common_types = {}
        
        for anime in self.completed_anime:
            all_genres.update(anime.genres)
            all_studios.update(anime.studios)
            common_types[anime.type_] = common_types.get(anime.type_, 0) + 1
        
        self.user_profile = {
            "genres": all_genres,
            "studios": all_studios,
            "avg_episodes": avg_episodes,
            "most_common_type": max(common_types, key=common_types.get) if common_types else "TV",
            "avg_user_score": np.mean([a.my_score for a in self.completed_anime]),
            "avg_mal_score": np.mean([a.mal_score for a in self.completed_anime]),
        }
    
    def _similarity_genre(self, anime: AnimeMetadata) -> float:
        """Compute genre similarity using Jaccard index."""
        return JaccardSimilarity.similarity(self.user_profile["genres"], anime.genres)
    
    def _similarity_user_score(self, anime: AnimeMetadata) -> float:
        """Compute similarity based on MAL score relative to user's average rated score."""
        if not self.user_profile:
            return 0.0
        
        user_avg = self.user_profile["avg_user_score"]
        # Higher similarity if MAL score aligns with user's preferences
        score_diff = abs(anime.mal_score - user_avg)
        return max(0.0, 1.0 - (score_diff / 10.0))  # Normalize to [0, 1]
    
    def _similarity_mal_score(self, anime: AnimeMetadata) -> float:
        """Normalize MAL score to [0, 1]."""
        return anime.mal_score / 10.0
    
    def _similarity_studio(self, anime: AnimeMetadata) -> float:
        """Compute studio overlap."""
        return JaccardSimilarity.similarity(self.user_profile["studios"], anime.studios)
    
    def _similarity_type(self, anime: AnimeMetadata) -> float:
        """Match anime type preference."""
        return 1.0 if anime.type_ == self.user_profile["most_common_type"] else 0.5
    
    def _similarity_episodes(self, anime: AnimeMetadata) -> float:
        """Compute similarity based on episode count."""
        user_avg_episodes = self.user_profile["avg_episodes"]
        if user_avg_episodes == 0:
            return 0.5
        
        ep_ratio = anime.episodes / user_avg_episodes
        # Prefer similar episode counts (ratio close to 1)
        return 1.0 / (1.0 + abs(ep_ratio - 1.0))
    
    def _similarity_recency(self, anime: AnimeMetadata) -> float:
        """Compute recency score based on release date (newer = higher score)."""
        if not anime.aired_from or not self.reference_date:
            return 0.5  # Neutral score if date unavailable
        
        try:
            from datetime import datetime
            aired_date = datetime.strptime(anime.aired_from, "%Y-%m-%d")
            years_old = (self.reference_date - aired_date).days / 365.25
            
            # Decay function: newer anime get higher scores
            # At decay_years, score = 0.5; older get lower scores
            # Within 1 year: score ≈ 0.9-1.0
            score = 1.0 / (1.0 + (years_old / self.recency_decay_years))
            return max(0.0, min(1.0, score))
        except (ValueError, AttributeError):
            return 0.5  # Neutral if date parsing fails
    
    def compute_similarity_score(self, anime: AnimeMetadata) -> float:
        """
        Compute comprehensive similarity score for an anime.
        
        Returns normalized score in [0, 1].
        """
        if not self.user_profile:
            raise ValueError("User profile not built. Call build_profile() first.")
        
        score = (
            self.weight_genre * self._similarity_genre(anime) +
            self.weight_user_score * self._similarity_user_score(anime) +
            self.weight_mal_score * self._similarity_mal_score(anime) +
            self.weight_studio * self._similarity_studio(anime) +
            self.weight_type * self._similarity_type(anime) +
            self.weight_episodes * self._similarity_episodes(anime) +
            self.weight_recency * self._similarity_recency(anime)
        )
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def recommend(
        self,
        candidate_anime: List[AnimeMetadata],
        n_recommendations: int = 10,
        exclude_watched: bool = True,
        min_score: float = 0.0,
    ) -> List[Tuple[AnimeMetadata, float]]:
        """
        Generate recommendations from a pool of candidate anime.
        
        Args:
            candidate_anime: List of candidate anime to recommend from
            n_recommendations: Number of recommendations to return
            exclude_watched: Whether to exclude already watched anime
            min_score: Minimum similarity score threshold
            
        Returns:
            List of (AnimeMetadata, similarity_score) tuples, sorted by score descending
        """
        if not self.user_profile:
            raise ValueError("User profile not built. Call build_profile() first.")
        
        # Filter candidates
        watched_ids = {a.mal_id for a in self.completed_anime}
        if exclude_watched:
            candidates = [a for a in candidate_anime if a.mal_id not in watched_ids]
        else:
            candidates = candidate_anime
        
        # Compute scores
        scored_anime = [
            (anime, self.compute_similarity_score(anime))
            for anime in candidates
        ]
        
        # Filter by minimum score and sort
        recommendations = [
            (anime, score) for anime, score in scored_anime
            if score >= min_score
        ]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def get_profile_summary(self) -> Dict:
        """Get a summary of the built user profile."""
        if not self.user_profile:
            return {}
        
        return {
            "num_completed_anime": len(self.completed_anime),
            "avg_user_score": self.user_profile["avg_user_score"],
            "avg_mal_score": self.user_profile["avg_mal_score"],
            "avg_episodes": self.user_profile["avg_episodes"],
            "most_common_type": self.user_profile["most_common_type"],
            "unique_genres": len(self.user_profile["genres"]),
            "unique_studios": len(self.user_profile["studios"]),
            "genres": sorted(self.user_profile["genres"]),
            "studios": sorted(self.user_profile["studios"]),
        }


def load_anime_from_json(filepath: str) -> List[AnimeMetadata]:
    """Load anime list from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return [AnimeMetadata.from_json(anime) for anime in data]


def save_recommendations_to_json(
    recommendations: List[Tuple[AnimeMetadata, float]],
    output_filepath: str,
) -> None:
    """Save recommendations to JSON file."""
    rec_data = [
        {
            "mal_id": anime.mal_id,
            "title": anime.title,
            "title_english": anime.title_english,
            "similarity_score": float(score),
            "mal_score": anime.mal_score,
            "genres": sorted(anime.genres),
            "type": anime.type_,
            "episodes": anime.episodes,
            "studios": sorted(anime.studios),
            "aired_from": anime.aired_from,
        }
        for anime, score in recommendations
    ]
    
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(rec_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Load anime data
    input_file = "data/input/myanimelist.json"
    output_file = "data/output/anime_recommendations.json"
    
    print("Loading anime data...")
    anime_list = load_anime_from_json(input_file)
    print(f"Loaded {len(anime_list)} anime")
    
    # Create recommender
    print("\nInitializing recommender...")
    recommender = ContentBasedRecommender(
        weight_genre=0.35,
        weight_user_score=0.25,
        weight_mal_score=0.15,
        weight_studio=0.1,
        weight_type=0.05,
        weight_episodes=0.05,
        min_user_rating=6.0,
    )
    
    # Build profile
    print("Building user profile...")
    recommender.build_profile(anime_list)
    profile = recommender.get_profile_summary()
    
    print(f"\nUser Profile:")
    print(f"  Completed anime (rating >= 6.0): {profile['num_completed_anime']}")
    print(f"  Average user score: {profile['avg_user_score']:.2f}")
    print(f"  Average MAL score: {profile['avg_mal_score']:.2f}")
    print(f"  Average episodes: {profile['avg_episodes']:.1f}")
    print(f"  Most common type: {profile['most_common_type']}")
    print(f"  Favorite genres: {', '.join(profile['genres'][:5])}")
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    recommendations = recommender.recommend(
        candidate_anime=anime_list,
        n_recommendations=10,
        exclude_watched=True,
        min_score=0.5,
    )
    
    print(f"\nTop 10 Recommendations:")
    for i, (anime, score) in enumerate(recommendations, 1):
        print(f"{i}. {anime.title} (Score: {score:.3f})")
        print(f"   Genres: {', '.join(anime.genres)}")
        print(f"   MAL Score: {anime.mal_score}")
    
    # Save recommendations
    print(f"\nSaving recommendations to {output_file}...")
    save_recommendations_to_json(recommendations, output_file)
    print("Done!")
