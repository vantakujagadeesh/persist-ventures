import requests
from typing import List, Dict, Optional
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
from fastapi import FastAPI, HTTPException
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class VideoRecommender:
    def __init__(self):
        self.flic_token = "flic_6e2d8d25dc29a4ddd382c2383a903cf4a688d1a117f6eb43b35a1e7fadbb84b8"
        self.headers = {"Flic-Token": self.flic_token}
        self.base_url = "https://api.socialverseapp.com"
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(hours=1)

    async def fetch_data_with_pagination(self, endpoint: str, params: Dict) -> List[Dict]:
        """Fetch paginated data from API with caching"""
        cache_key = f"{endpoint}_{str(params)}"
        
        # Check cache
        if cache_key in self.cache:
            if datetime.now() < self.cache_expiry[cache_key]:
                return self.cache[cache_key]
        
        all_data = []
        page = 1
        
        while True:
            params['page'] = page
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    params=params,
                    headers=self.headers
                )
                response.raise_for_status()
                data = response.json()
                
                if not data:  # No more data
                    break
                    
                all_data.extend(data)
                page += 1
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching data: {e}")
                break
        
        # Update cache
        self.cache[cache_key] = all_data
        self.cache_expiry[cache_key] = datetime.now() + self.cache_duration
        
        return all_data

    async def get_user_interactions(self, username: str) -> Dict:
        """Fetch all user interactions"""
        params = {
            "page_size": 1000,
            "resonance_algorithm": "resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"
        }
        
        viewed = await self.fetch_data_with_pagination("/posts/view", params)
        liked = await self.fetch_data_with_pagination("/posts/like", params)
        inspired = await self.fetch_data_with_pagination("/posts/inspire", params)
        rated = await self.fetch_data_with_pagination("/posts/rating", params)
        
        return {
            "viewed": viewed,
            "liked": liked,
            "inspired": inspired,
            "rated": rated
        }

    def calculate_user_preferences(self, interactions: Dict) -> Dict:
        """Calculate user preferences based on interactions"""
        preferences = defaultdict(float)
        
        # Weight different types of interactions
        weights = {
            "viewed": 1.0,
            "liked": 2.0,
            "inspired": 3.0,
            "rated": 2.5
        }
        
        for interaction_type, weight in weights.items():
            for interaction in interactions[interaction_type]:
                category_id = interaction.get("category_id")
                if category_id:
                    preferences[category_id] += weight
                    
        # Normalize preferences
        total = sum(preferences.values()) or 1
        return {k: v/total for k, v in preferences.items()}

    async def get_recommendations(
        self, 
        username: str, 
        category_id: Optional[str] = None, 
        mood: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Get personalized video recommendations"""
        try:
            # Get user interactions
            interactions = await self.get_user_interactions(username)
            user_preferences = self.calculate_user_preferences(interactions)
            
            # Fetch all posts
            all_posts = await self.fetch_data_with_pagination(
                "/posts/summary/get",
                {"page_size": 1000}
            )
            
            # Filter and score posts
            scored_posts = []
            viewed_posts = {post["id"] for post in interactions["viewed"]}
            
            for post in all_posts:
                if post["id"] in viewed_posts:
                    continue
                    
                score = 0
                
                # Category preference
                if category_id:
                    if post.get("category_id") == category_id:
                        score += 2.0
                else:
                    score += user_preferences.get(post.get("category_id"), 0)
                
                # Mood matching
                if mood and post.get("mood") == mood:
                    score += 1.0
                
                # Recency boost
                posted_date = datetime.fromisoformat(post.get("created_at", "2024-01-01"))
                days_old = (datetime.now() - posted_date).days
                recency_score = 1.0 / (1 + days_old/30)  # Decay over 30 days
                score += recency_score
                
                scored_posts.append((score, post))
            
            # Sort by score and return top recommendations
            scored_posts.sort(reverse=True, key=lambda x: x[0])
            return [post for score, post in scored_posts[:limit]]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise HTTPException(status_code=500, detail="Error generating recommendations")

# FastAPI endpoints
@app.get("/feed")
async def get_feed(
    username: str,
    category_id: Optional[str] = None,
    mood: Optional[str] = None
):
    recommender = VideoRecommender()
    recommendations = await recommender.get_recommendations(
        username=username,
        category_id=category_id,
        mood=mood
    )
    return recommendations
