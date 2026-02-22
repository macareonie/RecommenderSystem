"""
Main script to generate anime recommendations.
Integrates scraping candidates from Jikan API and generating recommendations.
"""

import sys
import json
from pathlib import Path
from src.recommender import (
    ContentBasedRecommender,
    load_anime_from_json,
    save_recommendations_to_json,
    AnimeMetadata,
)
from src.scraper import AnimeScraperManager


def main():
    """Main execution function."""
    
    # File paths
    input_file = Path("data/input/myanimelist.json")
    candidates_file = Path("data/candidates/anime_candidates.json")
    output_file = Path("data/output/anime_recommendations.json")
    
    # Validate input file
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found!")
        return 1
    
    print("=" * 70)
    print("ANIME RECOMMENDER SYSTEM (with Recency Bias)")
    print("=" * 70)
    
    # Load user's anime data
    print("\n[1/5] Loading your anime data...")
    try:
        user_anime = load_anime_from_json(str(input_file))
        print(f"      ✓ Loaded {len(user_anime)} anime from your list")
    except Exception as e:
        print(f"      ✗ Error loading data: {e}")
        return 1
    
    # Fetch candidate anime if not already cached
    print("\n[2/5] Fetching candidate anime...")
    candidates_data = None
    
    # Try released candidates first, then fallback to combined file
    released_file = Path("data/candidates/anime_candidates_released.json")
    if released_file.exists():
        print("      Using cached released candidates...")
        with open(released_file, "r", encoding="utf-8") as f:
            candidates_data = json.load(f)
        print(f"      ✓ Loaded {len(candidates_data)} cached released candidates")
    elif candidates_file.exists():
        print("      Using cached candidates...")
        with open(candidates_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
            # Handle both list and dict formats
            if isinstance(cached, dict) and "released" in cached:
                candidates_data = cached["released"]
            else:
                candidates_data = cached
        print(f"      ✓ Loaded {len(candidates_data)} cached candidates")
    else:
        print("      Fetching from Jikan API...")
        try:
            manager = AnimeScraperManager(output_dir=str(candidates_file.parent))
            sources = [
                {"type": "year", "year": 2020, "limit": 50},
                {"type": "upcoming", "limit": 50},
            ]
            result = manager.fetch_candidates(
                sources=sources,
                separate_upcoming=True
            )
            # Combine released and upcoming for recommendations
            candidates_data = result["released"] + result.get("upcoming", [])
            print(f"      ✓ Fetched {len(candidates_data)} total candidates")
        except Exception as e:
            print(f"      ✗ Error fetching candidates: {e}")
            print("      Note: Make sure 'requests' library is installed: pip install requests")
            return 1
    
    # Convert candidates to AnimeMetadata
    print("\n[3/5] Processing candidates...")
    try:
        candidate_anime = [AnimeMetadata.from_json(anime) for anime in candidates_data]
        print(f"      ✓ Processed {len(candidate_anime)} candidate anime")
    except Exception as e:
        print(f"      ✗ Error processing candidates: {e}")
        return 1
    
    # Create and configure recommender
    print("\n[4/5] Building user profile and generating recommendations...")
    recommend_config = {
        "weight_genre": 0.30,           # Genre similarity (Jaccard)
        "weight_user_score": 0.22,      # User rating alignment
        "weight_mal_score": 0.13,       # Community score credibility
        "weight_studio": 0.10,          # Studio preference
        "weight_type": 0.05,            # Type preference (TV/Movie)
        "weight_episodes": 0.05,        # Episode count similarity
        "weight_recency": 0.15,         # Release date recency bias
        "min_user_rating": 6.0,         # Minimum rating to build profile
        "recency_decay_years": 5.0,     # Decay half-life in years
    }
    
    try:
        recommender = ContentBasedRecommender(**recommend_config)
        recommender.build_profile(user_anime)
        profile = recommender.get_profile_summary()
        
        print(f"      ✓ Profile built from {profile['num_completed_anime']} rated anime")
        print(f"        - Avg user rating: {profile['avg_user_score']:.2f}/10")
        print(f"        - Avg MAL score: {profile['avg_mal_score']:.2f}/10")
        print(f"        - Avg episodes: {profile['avg_episodes']:.1f}")
        print(f"        - Preferred type: {profile['most_common_type']}")
        print(f"        - Unique genres: {profile['unique_genres']}")
        print(f"        - Top genres: {', '.join(profile['genres'][:5])}")
        
        # Generate recommendations
        recommendations = recommender.recommend(
            candidate_anime=candidate_anime,
            n_recommendations=10,
            exclude_watched=True,
            min_score=0.5,
        )
        
        if not recommendations:
            print("      ⚠ No recommendations found with current filters")
            return 0
        
        print(f"      ✓ Generated {len(recommendations)} recommendations")
        
    except Exception as e:
        print(f"      ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Display and save recommendations
    print("\n[5/5] Saving results...")
    try:
        print("\nTop 10 Anime Recommendations:")
        print("-" * 70)
        for i, (anime, score) in enumerate(recommendations, 1):
            print(f"{i:2}. {anime.title}")
            print(f"    Score: {score:.3f} | MAL: {anime.mal_score}/10 | Eps: {anime.episodes}")
            print(f"    Released: {anime.aired_from if anime.aired_from else 'Unknown'}")
            print(f"    Genres: {', '.join(anime.genres)}")
            if anime.studios:
                studios_str = ', '.join(list(anime.studios)[:2])
                if len(anime.studios) > 2:
                    studios_str += f", +{len(anime.studios)-2} more"
                print(f"    Studios: {studios_str}")
            print()
        
        # Save recommendations
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_recommendations_to_json(recommendations, str(output_file))
        print("-" * 70)
        print(f"✓ Recommendations saved to {output_file}")
        
    except Exception as e:
        print(f"      ✗ Error saving results: {e}")
        return 1
    
    print("\n" + "=" * 70)
    print("Recommendation complete!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
