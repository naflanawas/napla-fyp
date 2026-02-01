"""
MURMUR Backend API Server

FastAPI server providing calibration and prediction endpoints for breath-based AAC.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging
import traceback

from config import HOST, PORT, CORS_ORIGINS
from model import MSTCNEmbedder
from audio_processor import AudioProcessor, BreathDetector
from protonet import ProtoNet
from user_manager import UserManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="MURMUR API",
    description="AI-powered AAC system for breath pattern recognition",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
embedder = MSTCNEmbedder(device="cpu")
audio_processor = AudioProcessor()
breath_detector = BreathDetector()
protonet = ProtoNet()
user_manager = UserManager()


# ============= Response Models =============

class HealthResponse(BaseModel):
    status: str
    message: str

class CalibrateResponse(BaseModel):
    success: bool
    message: str
    intent: str
    sample_count: int

class PredictResponse(BaseModel):
    success: bool
    intent: str
    confidence: float
    distance: float
    is_confident: bool
    phrase: str
    all_distances: dict
    is_sequence_complete: bool = True # Default for atomic intents
    matched_sequence: Optional[str] = None # e.g. "short,long"

class IntentInfo(BaseModel):
    name: str
    sample_count: int
    phrase: str

class UserIntentsResponse(BaseModel):
    user_id: str
    intents: List[IntentInfo]
    sequences: Dict[str, str] = {} # "short,long" -> "Help"

class UserStatsResponse(BaseModel):
    user_id: str
    exists: bool
    num_intents: int
    total_samples: int
    intents: List[dict]


# ============= Startup / Shutdown =============

@app.on_event("startup")
async def startup():
    """Load saved prototypes on startup"""
    logger.info("Starting MURMUR API server...")
    
    # Load existing user data
    users = user_manager.list_users()
    for user_id in users:
        state = user_manager.load_prototypes(user_id)
        if state:
            protonet.load_state(state)
            logger.info(f"Loaded prototypes for user: {user_id}")
    
    logger.info(f"Server ready. Loaded {len(users)} user(s).")


# ============= Health Check =============

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="MURMUR API is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        message=f"Model loaded, {len(user_manager.list_users())} users configured"
    )


# ============= Calibration Endpoints =============

@app.post("/calibrate/{user_id}/{intent}", response_model=CalibrateResponse)
async def calibrate(
    user_id: str,
    intent: str,
    file: UploadFile = File(...),
    label: Optional[str] = Form(None), # New field for specific sample label
    phrase: Optional[str] = Form(None)
):
    """
    Add a calibration sample for a user's intent.
    
    - **user_id**: Unique identifier for the user.
    - **intent**: Intent name (e.g., "water", "help").
    - **file**: Audio file (WAV format recommended).
    - **label**: Optional label for the specific sample (e.g., "soft", "loud").
    - **phrase**: Optional phrase to speak when this intent is detected.
    """
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Process audio
        try:
            audio = audio_processor.load_audio_from_bytes(audio_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio format: {str(e)}")
        
        # Check for breath and validate duration with label
        if label in ['short', 'long']:
            is_valid, msg = breath_detector.validate_breath_duration(audio, label)
            if not is_valid:
                 raise HTTPException(status_code=400, detail=f"Validation failed: {msg}")
        elif not breath_detector.is_breath_detected(audio):
            # Fallback for other labels or if no label provided
            logger.warning(f"No breath detected in calibration for {user_id}/{intent}")
            raise HTTPException(status_code=400, detail="No breath detected in audio")
        
        # Extract features and embeddings
        features = audio_processor.process_audio(audio)
        
        # Get embedding (average if multiple windows)
        if features.shape[-1] > audio_processor.window_size:
            windows = audio_processor.get_windows_tensor(audio)
            embedding = embedder.extract_embedding(windows).mean(dim=0)
        else:
            embedding = embedder.extract_embedding(features.unsqueeze(0)).squeeze(0)
        
        # Combine intent and label for prototype storage
        prototype_key = f"{intent}_{label}" if label else intent
        
        # Add to prototypes
        protonet.add_sample(user_id, prototype_key, embedding)
        sample_count = protonet.get_sample_count(user_id, prototype_key)
        
        # Save phrase if provided, associate with the base intent
        if phrase:
            user_manager.set_intent_phrase(user_id, intent, phrase)
        elif not user_manager.get_intent_phrase(user_id, intent):
            # Set default phrase if no phrase exists for the base intent
            user_manager.set_intent_phrase(user_id, intent, f"I need {intent}")
        
        # Persist state
        user_manager.save_prototypes(user_id, protonet.get_state())
        
        logger.info(f"Calibration: {user_id}/{prototype_key} - {sample_count} samples")
        
        return CalibrateResponse(
            success=True,
            message=f"Added calibration sample for '{prototype_key}'",
            intent=intent, # Return base intent for consistency
            sample_count=sample_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Calibration error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/calibrate/{user_id}/{intent}")
async def delete_calibration_sample(
    user_id: str,
    intent: str,
    label: Optional[str] = Form(None) # Use Form for optional body parameter
):
    """
    Delete a specific calibration sample for a user's intent.
    
    - **user_id**: Unique identifier for the user.
    - **intent**: Intent name (e.g., "water", "help").
    - **label**: Optional label for the specific sample to delete. If not provided,
                 it will attempt to delete a sample associated with the base intent.
    """
    try:
        prototype_key = f"{intent}_{label}" if label else intent
        
        success = protonet.delete_sample(user_id, prototype_key)
        
        if success:
            # Persist state
            user_manager.save_prototypes(user_id, protonet.get_state())
            logger.info(f"Deleted calibration sample: {user_id}/{prototype_key}")
            return {"success": True, "message": f"Deleted calibration sample for '{prototype_key}'"}
        else:
            raise HTTPException(status_code=404, detail=f"Calibration sample for '{prototype_key}' not found for user '{user_id}'")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete calibration sample error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/map_sequence/{user_id}")
async def map_sequence(
    user_id: str,
    sequence: str = Form(...),  # e.g., "short,long"
    phrase: str = Form(...)
):
    """Map a sequence of breath tokens to a phrase"""
    seq_list = [s.strip() for s in sequence.split(",")]
    
    # Check for existing mapping
    existing_phrase = user_manager.get_phrase_for_sequence(user_id, seq_list)
    
    # get_phrase_for_sequence is robust (checks intents too), so we should check specifically the sequence map
    config = user_manager.load_config(user_id)
    seq_key = ",".join(seq_list)
    
    if "sequence_phrases" in config and seq_key in config["sequence_phrases"]:
        existing = config["sequence_phrases"][seq_key]
        if existing != phrase:
            # Strictly block saving if duplicate
            raise HTTPException(
                status_code=400, 
                detail=f"Sequence '{seq_key}' is already mapped to '{existing}'. Please delete it first."
            )
            
    user_manager.set_sequence_phrase(user_id, seq_list, phrase)
    return {"success": True, "sequence": seq_list, "phrase": phrase}

@app.post("/clear_buffer/{user_id}")
async def clear_buffer(user_id: str):
    """Clear the user's breath token buffer"""
    user_manager.clear_user_buffer(user_id)
    return {"success": true, "message": "Buffer cleared"}

@app.delete("/user/{user_id}/sequence")
async def delete_sequence(user_id: str, sequence: str = Form(...)):
    """Delete a specific mapped sequence"""
    config = user_manager.load_config(user_id)
    if "sequence_phrases" in config and sequence in config["sequence_phrases"]:
        del config["sequence_phrases"][sequence]
        user_manager.save_config(user_id, config)
        return {"success": True, "message": f"Sequence '{sequence}' deleted"}
    return {"success": False, "message": f"Sequence '{sequence}' not found"}

# ============= Prediction Endpoints =============

@app.post("/predict/{user_id}", response_model=PredictResponse)
async def predict(
    user_id: str,
    file: UploadFile = File(...),
    aggregation: Optional[str] = Form("voting")
):
    """
    Predict intent from breath audio and handle sequences.
    """
    try:
        # 1. Read and process audio
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        try:
            audio = audio_processor.load_audio_from_bytes(audio_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio format: {str(e)}")
        
        # 2. Extract features and embeddings
        features = audio_processor.process_audio(audio)
        if features.shape[-1] > audio_processor.window_size:
            windows = audio_processor.get_windows_tensor(audio)
            embeddings = embedder.extract_embedding(windows)
            result = protonet.predict_with_aggregation(user_id, embeddings, method=aggregation)
        else:
            embedding = embedder.extract_embedding(features.unsqueeze(0)).squeeze(0)
            result = protonet.predict(user_id, embedding)
        
        # 3. Handle Sequence Logic
        # Result intent will be like "short" or "water_short"
        # We extract the 'label' part which is what we buffer
        token = result.intent.split("_")[-1] if "_" in result.intent else result.intent
        
        if result.is_confident:
            user_manager.append_to_user_buffer(user_id, token)
            current_buffer = user_manager.get_user_buffer(user_id)
            config = user_manager.load_config(user_id)
            all_seqs = list(config.get("sequence_phrases", {}).keys())
            
            phrase_to_return = ""
            is_complete = True
            matched_seq_str = ",".join(current_buffer)
            
            # Check for sequence matches (longest match first)
            for i in range(len(current_buffer), 0, -1):
                sub_seq_list = current_buffer[-i:]
                sub_seq_str = ",".join(sub_seq_list).lower()
                phrase = user_manager.get_phrase_for_sequence(user_id, sub_seq_list)
                
                if phrase:
                    # Check if this sub-sequence is a prefix of any LONGER sequence in the library
                    # We only buffer if the buffer size is reasonable
                    is_prefix = any(s.startswith(sub_seq_str + ",") for s in all_seqs)
                    
                    if is_prefix and len(current_buffer) < 5:
                        phrase_to_return = "..." # Silent buffering
                        is_complete = False
                        matched_seq_str = sub_seq_str
                        logger.info(f"Buffered {sub_seq_str} (possible prefix)")
                        break
                    else:
                        phrase_to_return = phrase
                        is_complete = True
                        matched_seq_str = sub_seq_str
                        user_manager.clear_user_buffer(user_id)
                        logger.info(f"Matched sequence: {sub_seq_str} -> {phrase}")
                        break
            else:
                # No sequence match found in library, use the best intent phrase we have
                # This ensures single calibrated intents still speak if not mapped as sequences
                phrase_to_return = user_manager.get_intent_phrase(user_id, result.intent)
                is_complete = True
                matched_seq_str = token
                logger.info(f"No sequence match, fallback to intent: {result.intent} -> {phrase_to_return}")
        else:
            phrase_to_return = "Unknown"
            is_complete = False
            matched_seq_str = None
            logger.warning(f"Low confidence prediction for {user_id}: {result.intent} ({result.confidence})")
            
        logger.info(f"Prediction: {user_id} -> {result.intent} (complete={is_complete}, seq={matched_seq_str}, buffer: {user_manager.get_user_buffer(user_id)})")
        
        return PredictResponse(
            success=True,
            intent=result.intent,
            confidence=result.confidence,
            distance=result.distance,
            is_confident=result.is_confident,
            phrase=phrase_to_return,
            all_distances=result.all_distances,
            is_sequence_complete=is_complete,
            matched_sequence=matched_seq_str
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= User Management Endpoints =============

@app.get("/user/{user_id}/intents", response_model=UserIntentsResponse)
async def get_user_intents(user_id: str):
    """Get all calibrated intents for a user"""
    intents = protonet.get_user_intents(user_id)
    config = user_manager.load_config(user_id)
    
    intent_infos = []
    for intent in intents:
        intent_infos.append(IntentInfo(
            name=intent,
            sample_count=protonet.get_sample_count(user_id, intent),
            phrase=user_manager.get_intent_phrase(user_id, intent)
        ))
    
    return UserIntentsResponse(
        user_id=user_id,
        intents=intent_infos,
        sequences=config.get("sequence_phrases", {})
    )

@app.get("/user/{user_id}/stats", response_model=UserStatsResponse)
async def get_user_stats(user_id: str):
    """Get detailed statistics for a user"""
    stats = user_manager.get_user_stats(user_id)
    return UserStatsResponse(
        user_id=user_id,
        exists=stats["exists"],
        num_intents=stats["num_intents"],
        total_samples=stats["total_samples"],
        intents=stats["intents"]
    )

@app.delete("/user/{user_id}/intent/{intent}")
async def delete_intent(user_id: str, intent: str):
    """Delete an intent for a user"""
    success = protonet.delete_intent(user_id, intent)
    
    if success:
        user_manager.save_prototypes(user_id, protonet.get_state())
        return {"success": True, "message": f"Deleted intent '{intent}'"}
    
    raise HTTPException(status_code=404, detail=f"Intent '{intent}' not found")

@app.delete("/user/{user_id}")
async def delete_user(user_id: str):
    """Delete all data for a user"""
    protonet.delete_user(user_id)
    user_manager.delete_user(user_id)
    return {"success": True, "message": f"Deleted user '{user_id}'"}

@app.get("/users")
async def list_users():
    """List all registered users"""
    users = user_manager.list_users()
    return {"users": users}


# ============= Intent Phrase Management =============

@app.put("/user/{user_id}/intent/{intent}/phrase")
async def update_intent_phrase(
    user_id: str, 
    intent: str, 
    phrase: str = Form(...)
):
    """Update the phrase for an intent"""
    if intent not in protonet.get_user_intents(user_id):
        raise HTTPException(status_code=404, detail=f"Intent '{intent}' not found")
    
    user_manager.set_intent_phrase(user_id, intent, phrase)
    return {"success": True, "intent": intent, "phrase": phrase}


# ============= Run Server =============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
