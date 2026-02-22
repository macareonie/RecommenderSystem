"""
Anime scraper using Jikan API to fetch anime candidates for recommendations.
Supports fetching by release year, upcoming, popular, airing, and genre-specific anime.
"""

import requests
import json
import time
from typing import List, Dict, Optional
from pathlib import Path


class JikanAPI:
    """Wrapper for Jikan API (unofficial MyAnimeList API)."""
    
    BASE_URL = "https://api.jikan.moe/v4"
    REQUEST_DELAY = 0.5
    
    def __init__(self, request_delay: float = 0.5):
        self.request_delay = request_delay
        self.last_request_time = 0
    
    def _rate_limit(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()
    
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        self._rate_limit()
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    
    def _fetch_paginated(self, endpoint: str, params: Dict, limit: int) -> List[Dict]:
        all_anime = []
        page = 1
        
        while len(all_anime) < limit:
            params["page"] = page
            params["limit"] = min(25, limit - len(all_anime))
            
            response = self._get(endpoint, params=params)
            
            if "data" not in response or not response["data"]:
                break
            
            all_anime.extend(response["data"])
            page += 1
            
            if not response.get("pagination", {}).get("has_next_page", False):
                break
        
        return all_anime[:limit]
    
    def get_anime_by_year(self, year: int, limit: int = 25) -> List[Dict]:
        print(f"Fetching anime from {year} onwards (limit: {limit})...")
        
        all_anime = []
        page = 1
        
        while len(all_anime) < limit:
            params = {
                "min_score": 1,
                "page": page,
                "limit": min(25, limit - len(all_anime)),
                "order_by": "score",
                "sort": "desc",
            }
            
            response = self._get("/anime", params=params)
            
            if "data" not in response or not response["data"]:
                break
            
            for anime in response["data"]:
                if len(all_anime) >= limit:
                    break
                
                aired = anime.get("aired", {})
                aired_from = aired.get("from")
                
                if aired_from:
                    anime_year = int(aired_from[:4])
                    if anime_year >= year:
                        all_anime.append(anime)
            
            page += 1
            
            if not response.get("pagination", {}).get("has_next_page", False):
                break
        
        return all_anime[:limit]
    
    def get_upcoming_anime(self, limit: int = 25) -> List[Dict]:
        print(f"Fetching upcoming anime (limit: {limit})...")
        return self._fetch_paginated("/anime", {"status": "upcoming"}, limit)
    
    def get_airing_anime(self, limit: int = 25) -> List[Dict]:
        print(f"Fetching airing anime (limit: {limit})...")
        return self._fetch_paginated("/anime", {"status": "airing"}, limit)
    
    def get_popular_anime(self, limit: int = 25) -> List[Dict]:
        print(f"Fetching popular anime (limit: {limit})...")
        return self._fetch_paginated("/top/anime", {}, limit)
    
    def get_anime_by_genre(self, genre_id: int, limit: int = 25) -> List[Dict]:
        print(f"Fetching anime by genre ID {genre_id} (limit: {limit})...")
        return self._fetch_paginated("/anime", {"genres": genre_id}, limit)
    
    def get_seasonal_anime(self, year: int, season: str, limit: int = 25) -> List[Dict]:
        season = season.lower()
        if season not in ["winter", "spring", "summer", "fall"]:
            raise ValueError(f"Invalid season: {season}")
        
        print(f"Fetching {season.capitalize()} {year} anime (limit: {limit})...")
        response = self._get(f"/seasons/{year}/{season}", params={"limit": limit})
        
        if "data" not in response:
            return []
        
        return response["data"][:limit]


class AnimeNormalizer:
    """Normalize Jikan API response to standard format."""
    
    @staticmethod
    def normalize(jikan_anime: Dict) -> Dict:
        return {
            "mal_id": str(jikan_anime.get("mal_id", "")),
            "title": jikan_anime.get("title", ""),
            "title_english": jikan_anime.get("title_english") or jikan_anime.get("title", ""),
            "type": jikan_anime.get("type", ""),
            "episodes": jikan_anime.get("episodes") or 0,
            "my_status": "Plan to Watch",
            "my_score": 0,
            "my_watched_episodes": 0,
            "my_start_date": "0000-00-00",
            "my_finish_date": "0000-00-00",
            "my_comments": None,
            "my_tags": None,
            "synopsis": jikan_anime.get("synopsis", ""),
            "mal_score": jikan_anime.get("score") or 0.0,
            "genres": [
                {
                    "mal_id": g.get("mal_id", 0),
                    "type": "anime",
                    "name": g.get("name", ""),
                    "url": g.get("url", ""),
                }
                for g in jikan_anime.get("genres", [])
            ],
            "image_url": jikan_anime.get("images", {}).get("jpg", {}).get("image_url", ""),
            "aired_from": AnimeNormalizer._extract_aired_date(jikan_anime),
            "studios": [
                {
                    "mal_id": s.get("mal_id", 0),
                    "type": "anime",
                    "name": s.get("name", ""),
                    "url": s.get("url", ""),
                }
                for s in jikan_anime.get("studios", [])
            ],
        }
    
    @staticmethod
    def _extract_aired_date(jikan_anime: Dict) -> str:
        aired = jikan_anime.get("aired", {})
        aired_from = aired.get("from")
        
        if aired_from:
            return aired_from.split("T")[0]
        return ""


class AnimeScraperManager:
    """Manager for scraping and aggregating anime data."""
    
    def __init__(self, output_dir: str = "data/candidates"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jikan = JikanAPI()
    
    def fetch_candidates(
        self,
        sources: List[Dict] = None,
        separate_upcoming: bool = True,
    ) -> Dict[str, List[Dict]]:
        """
        Fetch anime candidates from multiple sources.
        
        Args:
            sources: List of source configurations
            separate_upcoming: If True, separate upcoming from released anime
            
        Returns:
            Dictionary with "released" and "upcoming" lists
        """
        if sources is None:
            sources = [
                {"type": "year", "year": 2020, "limit": 200},
                {"type": "upcoming", "limit": 50},
            ]
        
        released_candidates = []
        upcoming_candidates = []
        seen_ids = set()
        
        print(f"\nFetching anime candidates from {len(sources)} sources...")
        print("=" * 60)
        
        for source in sources:
            source_type = source.get("type", "").lower()
            limit = source.get("limit", 25)
            
            try:
                if source_type == "year":
                    year = source.get("year", 2020)
                    anime_list = self.jikan.get_anime_by_year(year, limit=limit)
                elif source_type == "upcoming":
                    anime_list = self.jikan.get_upcoming_anime(limit=limit)
                elif source_type == "airing":
                    anime_list = self.jikan.get_airing_anime(limit=limit)
                elif source_type == "popular":
                    anime_list = self.jikan.get_popular_anime(limit=limit)
                elif source_type == "seasonal":
                    year = source.get("year", 2025)
                    season = source.get("season", "winter")
                    anime_list = self.jikan.get_seasonal_anime(year, season, limit=limit)
                elif source_type == "genre":
                    genre_id = source.get("genre_id", 1)
                    anime_list = self.jikan.get_anime_by_genre(genre_id, limit=limit)
                else:
                    print(f"⚠ Unknown source type: {source_type}")
                    continue
                
                for anime in anime_list:
                    mal_id = str(anime.get("mal_id", ""))
                    if mal_id and mal_id not in seen_ids:
                        seen_ids.add(mal_id)
                        normalized = AnimeNormalizer.normalize(anime)
                        
                        if separate_upcoming and source_type == "upcoming":
                            upcoming_candidates.append(normalized)
                        else:
                            released_candidates.append(normalized)
                
                print(f"✓ Added {len(anime_list)} from {source_type}")
                
            except Exception as e:
                print(f"✗ Error fetching {source_type}: {e}")
        
        print("=" * 60)
        print(f"Total released candidates: {len(released_candidates)}")
        print(f"Total upcoming candidates: {len(upcoming_candidates)}")
        
        results = {"released": released_candidates}
        
        if separate_upcoming and upcoming_candidates:
            results["upcoming"] = upcoming_candidates
            
            with open(self.output_dir / "anime_candidates_released.json", "w", encoding="utf-8") as f:
                json.dump(released_candidates, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved released to anime_candidates_released.json")
            
            with open(self.output_dir / "anime_candidates_upcoming.json", "w", encoding="utf-8") as f:
                json.dump(upcoming_candidates, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved upcoming to anime_candidates_upcoming.json")
        else:
            with open(self.output_dir / "anime_candidates.json", "w", encoding="utf-8") as f:
                json.dump(released_candidates, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved to anime_candidates.json")
        
        return results


JIKAN_GENRES = {
    1: "Action",
    2: "Adventure",
    4: "Comedy",
    8: "Drama",
    9: "Fantasy",
    10: "Game",
    14: "Horror",
    22: "Romance",
    24: "School",
    25: "Sci-Fi",
    27: "Shounen",
    41: "Thriller",
    43: "Shoujo",
    44: "Kids",
    52: "Psychological",
}


if __name__ == "__main__":
    manager = AnimeScraperManager()
    sources = [
        {"type": "year", "year": 2020, "limit": 500},
        {"type": "upcoming", "limit": 50},
    ]
    candidates = manager.fetch_candidates(sources=sources, separate_upcoming=True)
    
    print(f"\nFirst 3 released candidates:")
    for anime in candidates["released"][:3]:
        print(f"  - {anime['title']} (MAL Score: {anime['mal_score']})")
    
    if "upcoming" in candidates and candidates["upcoming"]:
        print(f"\nFirst 3 upcoming candidates:")
        for anime in candidates["upcoming"][:3]:
            print(f"  - {anime['title']} (MAL Score: {anime['mal_score']})")