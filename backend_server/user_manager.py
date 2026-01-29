"""
User Data Manager for MURMUR

Handles persistent storage of user prototypes and configurations.
"""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from config import USER_DATA_DIR


class UserManager:
    """Manages user data persistence"""
    
    def __init__(self, data_dir: Path = USER_DATA_DIR):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_user_dir(self, user_id: str) -> Path:
        """Get user-specific directory"""
        user_dir = self.data_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    
    def _get_prototypes_path(self, user_id: str) -> Path:
        """Get path to user's prototypes file"""
        return self._get_user_dir(user_id) / "prototypes.json"
    
    def _get_config_path(self, user_id: str) -> Path:
        """Get path to user's config file"""
        return self._get_user_dir(user_id) / "config.json"
    
    def user_exists(self, user_id: str) -> bool:
        """Check if user data exists"""
        return self._get_prototypes_path(user_id).exists()
    
    def save_prototypes(self, user_id: str, state: dict) -> None:
        """Save user prototypes to disk"""
        path = self._get_prototypes_path(user_id)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_prototypes(self, user_id: str) -> Optional[dict]:
        """Load user prototypes from disk"""
        path = self._get_prototypes_path(user_id)
        if not path.exists():
            return None
        with open(path, 'r') as f:
            return json.load(f)
    
    def save_config(self, user_id: str, config: dict) -> None:
        """Save user configuration"""
        path = self._get_config_path(user_id)
        config['updated_at'] = datetime.now().isoformat()
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, user_id: str) -> dict:
        """Load user configuration"""
        path = self._get_config_path(user_id)
        if not path.exists():
            return self._default_config()
        with open(path, 'r') as f:
            return json.load(f)
    
    def _default_config(self) -> dict:
        """Default user configuration"""
        return {
            "intent_phrases": {
                # intent_name: spoken_phrase (for backward compatibility)
            },
            "sequence_phrases": {
                # "short,long": "Help"
            },
            "confidence_threshold": 0.7,
            "aggregation_method": "voting",
            "created_at": datetime.now().isoformat()
        }

    def set_sequence_phrase(self, user_id: str, sequence: List[str], phrase: str) -> None:
        """Map a sequence of breath tokens to a phrase"""
        config = self.load_config(user_id)
        if "sequence_phrases" not in config:
            config["sequence_phrases"] = {}
        
        seq_key = ",".join(sequence)
        config["sequence_phrases"][seq_key] = phrase
        self.save_config(user_id, config)

    def get_phrase_for_sequence(self, user_id: str, sequence: List[str]) -> Optional[str]:
        """
        Get phrase for a sequence of tokens.
        Highly robust: checks both sequence_phrases and intent_phrases (for single tokens).
        """
        config = self.load_config(user_id)
        seq_phrases = config.get("sequence_phrases", {})
        intent_phrases = config.get("intent_phrases", {})
        
        seq_key = ",".join(sequence).lower()
        
        # 1. Check direct sequence mapping
        if seq_key in seq_phrases:
            return seq_phrases[seq_key]
        
        # 2. For single-token sequences, check intent_phrases fallback
        if len(sequence) == 1:
            token = sequence[0].lower()
            if token in intent_phrases:
                return intent_phrases[token]
            
            # Check with/without breath_ prefix
            breath_token = f"breath_{token}" if not token.startswith("breath_") else token
            if breath_token in intent_phrases:
                return intent_phrases[breath_token]
                
            raw_token = token.replace("breath_", "")
            if raw_token in intent_phrases:
                return intent_phrases[raw_token]
        
        return None

    # Simple in-memory buffer for user sessions (reset on startup)
    _user_buffers: Dict[str, List[str]] = {}

    def get_user_buffer(self, user_id: str) -> List[str]:
        return self._user_buffers.get(user_id, [])

    def append_to_user_buffer(self, user_id: str, token: str) -> None:
        if user_id not in self._user_buffers:
            self._user_buffers[user_id] = []
        self._user_buffers[user_id].append(token)
        # Limit buffer size to prevent memory leaks
        if len(self._user_buffers[user_id]) > 5:
            self._user_buffers[user_id].pop(0)

    def clear_user_buffer(self, user_id: str) -> None:
        self._user_buffers[user_id] = []
    
    def set_intent_phrase(self, user_id: str, intent: str, phrase: str) -> None:
        """Set the phrase to speak for an intent"""
        config = self.load_config(user_id)
        if "intent_phrases" not in config:
            config["intent_phrases"] = {}
        config["intent_phrases"][intent] = phrase
        self.save_config(user_id, config)
    
    def get_intent_phrase(self, user_id: str, intent: str) -> str:
        """Get the phrase to speak for an intent"""
        config = self.load_config(user_id)
        phrases = config.get("intent_phrases", {})
        if intent in phrases:
            return phrases[intent]
        
        # Better defaults for building blocks
        if intent == "breath_short": return "Short Breath"
        if intent == "breath_long": return "Long Breath"
        
        clean_intent = intent.replace("breath_", "").replace("_", " ").title()
        return clean_intent
    
    def get_all_intent_phrases(self, user_id: str) -> Dict[str, str]:
        """Get all intent phrases for a user"""
        config = self.load_config(user_id)
        return config.get("intent_phrases", {})
    
    def list_users(self) -> List[str]:
        """List all registered users"""
        users = []
        for item in self.data_dir.iterdir():
            if item.is_dir():
                users.append(item.name)
        return users
    
    def delete_user(self, user_id: str) -> bool:
        """Delete all user data"""
        user_dir = self._get_user_dir(user_id)
        if user_dir.exists():
            shutil.rmtree(user_dir)
            return True
        return False
    
    def get_user_stats(self, user_id: str) -> dict:
        """Get statistics about a user"""
        state = self.load_prototypes(user_id)
        config = self.load_config(user_id)
        
        if state is None:
            return {
                "exists": False,
                "num_intents": 0,
                "total_samples": 0,
                "intents": []
            }
        
        prototypes = state.get("prototypes", {}).get(user_id, {})
        samples = state.get("samples", {}).get(user_id, {})
        
        intent_stats = []
        total_samples = 0
        
        for intent in prototypes.keys():
            sample_count = len(samples.get(intent, []))
            total_samples += sample_count
            intent_stats.append({
                "name": intent,
                "sample_count": sample_count,
                "phrase": self.get_intent_phrase(user_id, intent)
            })
        
        return {
            "exists": True,
            "num_intents": len(prototypes),
            "total_samples": total_samples,
            "intents": intent_stats,
            "config": config
        }
