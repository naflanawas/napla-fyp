# USER JOURNEY 1(TECH)

### **Actors**

* Patient (breath input)  
* Caregiver (assists, confirms)  
* MURMUR System (AI pipeline)

---

### **Stage 1: System Setup & Calibration**

**Input**

* Breath audio samples per intent (few-shot)

**System Actions**

* Audio preprocessing (normalisation)  
* Window segmentation (1024 frames)  
* Global MS-TCN model converts windows → embeddings  
* Per-intent prototypes created (mean embeddings)  
* Dynamic-K adapts to available samples

**Output**

* User-specific prototype set stored  
* No model retraining performed

---

### **Stage 2: Passive Listening (Runtime)**

**Input**

* Live microphone audio

**System Actions**

* Continuous or triggered audio capture  
* Breath detection  
* Window-level segmentation

**Output**

* Stream of breath windows ready for inference

---

### **Stage 3: Feature Learning**

**Input**

* Breath windows

**System Actions**

* Each window passed through frozen MS-TCN  
* Embedding vectors generated

**Output**

* Window-level embeddings (representation space)

---

### **Stage 4: Personalised Decision (ProtoNet)**

**Input**

* Window embedding \+ user prototypes

**System Actions**

* Compute distances to each prototype  
* Select closest prototype  
* Compute confidence margin (d₂ − d₁)

**Output**

* Predicted intent  
* Confidence score

---

### **Stage 5: Explainability & Validation**

**Input**

* Prediction \+ confidence

**System Actions**

* Flag low-confidence cases  
* Log prototype distances  
* Aggregate window decisions (optional)

**Output**

* Explainable, uncertainty-aware intent prediction

---

### **Stage 6: Output Delivery**

**Input**

* Confirmed intent

**System Actions**

* Map intent → AAC phrase/command  
* Send to UI / speech output

**Output**

* Spoken or displayed communication

# USER JOURNEY 2

### **Actors**

* **Patient** (produces breath input)  
* **Caregiver** (assists, guides, confirms)  
* **MURMUR System** (listens, suggests, supports communication)  
  ---

  ## **Stage 1: Assisted System Setup**

**Input**

* Caregiver opens the MURMUR application  
* Patient is positioned comfortably

**Caregiver Actions**

* Follows system guidance for initial setup  
* Explains the process calmly to the patient  
* Ensures patient is relaxed and ready

**System Actions**

* Provides step-by-step setup instructions  
* Prepares the system for breath calibration

**Output**

* System ready for breath learning  
* Patient comfortable and supported  
  ---

  ## **Stage 2: Breath Learning (Calibration)**

**Input**

* Short puff and long puff breath attempts from the patient

**Caregiver Actions**

* Guides the patient to produce a few short and long puffs  
* Confirms correct attempts in the app  
* Repeats if the patient needs more time

**System Actions**

* Records breath attempts  
* Learns how *this specific patient* performs short and long puffs  
* Stores breath patterns securely

**Output**

* Patient-specific breath patterns stored  
* No long training or effort required  
  ---

  ## **Stage 3: Mapping Breaths to Messages**

**Input**

* Caregiver-selected phrases (e.g., “Help”, “Water”, “Yes”, “No”)

**Caregiver Actions**

* Maps:  
  * short puff → chosen message  
  * long puff → chosen message  
* Reviews mappings and saves them

**System Actions**

* Stores personalised breath-to-message mappings  
* Confirms successful configuration

**Output**

* Personalised communication commands ready  
  ---

  ## **Stage 4: Passive Listening Mode**

**Input**

* Background environment (no active interaction)

**System Actions**

* Switches to passive listening  
* Listens continuously for breath signals  
* Does not require button presses or movement

**Patient Experience**

* Can rest normally  
* Does not need to prepare or move

**Output**

* System ready to detect communication attempts at any time  
  ---

  ## **Stage 5: Everyday Communication**

**Input**

* Patient produces a breath when they want to communicate

**System Actions**

* Detects the breath pattern  
* Suggests a message with a confidence indicator

**Caregiver Actions**

* Sees the suggested message  
* Confirms if correct  
* Cancels if unsure

**Output**

* Confirmed message delivered safely  
  ---

  ## **Stage 6: Feedback & Ongoing Support**

**Input**

* System confidence feedback  
* Caregiver observations

**Caregiver Actions**

* Notices when the system is confident or unsure  
* Adds more examples if needed  
* Adjusts message mappings if required

**System Actions**

* Improves understanding over time  
* Maintains stable behaviour

**Output**

* Reliable, personalised communication over long-term use  
* 

# Context.md

# **MURMUR: Project Context & Architecture**

## **1\. Project Overview**

**MURMUR** is an AI-powered Augmentative and Alternative Communication (AAC) system designed for individuals with severe motor impairments. It translates specific breathing patterns (sniffs, exhales, etc.) into spoken phrases (e.g., "I need water", "Help").

**Key Differentiation:**

* **Input:** Raw respiratory audio (breath sounds).  
* **Processing:** Window-level classification (1024 frames) using a frozen Global Model \+ Few-Shot Prototypical Networks.  
* **Personalization:** The system adapts to each user's unique breath signature without retraining the core deep learning model (Dynamic-K Adaptation).

---

## **2\. System Architecture**

### **High-Level Data Flow**

Microphone (Mobile) \-\> WAV File \-\> API Server (Python) \-\> Windowing \-\> Global Model (MS-TCN) \-\> Embedding \-\> ProtoNet Matcher \-\> Intent \-\> Mobile UI

### **Component 1: The "Brain" (Backend API)**

* **Tech Stack:** Python, FastAPI, PyTorch, Librosa.  
* **Core Model:** MS-TCN (Multi-Stage Temporal Convolutional Network).  
  * *Input:* Audio Window (1024 frames).  
  * *Output:* 128-dimensional Embedding Vector.  
* **Personalization Logic:**  
  * *Calibration:* User provides $N$ breaths for an intent. System computes the "Prototype" (mean vector).  
  * *Inference:* Incoming breath is compared to Prototypes using Euclidean Distance.  
* **Endpoints:**  
  * POST /calibrate/{user\_id}/{intent}: Uploads audio to create/update a prototype.  
  * POST /predict/{user\_id}: Uploads audio to get a prediction.

### **Component 2: The "Body" (Frontend App)**

* **Tech Stack:** Flutter (Dart).  
* **Key Dependencies:**  
  * record: For raw audio capture.  
  * http: For API communication.  
  * path\_provider: For file storage.  
  * flutter\_animate / custom\_painter: For real-time breath visualization (Waveform).

---

## **3\. End-to-End Pipeline (Detailed)**

### **Phase A: Setup (Calibration)**

1. **User Action:** Caregiver selects "Add Command" \-\> "Water".  
2. **Instruction:** "Breathe 'Short-Short-Long' now."  
3. **App:** Records calibration\_water.wav and sends to /calibrate.  
4. **Server:**  
   * Loads audio.  
   * Splits into windows.  
   * Runs Global Model \-\> Gets Embeddings.  
   * Calculates Mean Vector \-\> Saves as Prototype\_Water for this user.  
5. **Feedback:** App shows "Command Learned."

### **Phase B: Usage (Real-Time Inference)**

1. **User Action:** Patient breathes.  
2. **App:**  
   * Continually listens (Passive Mode) OR listens on trigger.  
   * Detects audio energy \> Threshold.  
   * Records for fixed duration (e.g., 2 seconds) or until silence.  
   * Saves input.wav.  
3. **Network:** App POSTs input.wav to /predict.  
4. **Server:**  
   * Extracts embeddings for the new breath.  
   * Calculates distance to Prototype\_Water, Prototype\_Help, etc.  
   * Finds closest match (e.g., "Water" distance \= 0.2, "Help" distance \= 1.5).  
   * *Confidence Check:* If distance \< Threshold (0.5), return "Water". Else return "Unknown".  
5. **App:**  
   * Receives {"intent": "Water", "confidence": "High"}.  
   * **TTS:** Speaks "I need water."  
   * **Visual:** Shows confirmation icon.

---

## **4\. Technical Constraints & Constants**

* **Sample Rate:** 16,000 Hz (Must match training data).  
* **Window Size:** 1024 frames (approx 64ms).  
* **Audio Format:** PCM WAV (Mono).  
* **Server URL:**  
  * *Emulator:* http://10.0.2.2:8000  
  * *Physical Device:* http://192.168.X.X:8000 (Local IP).

## **5\. Directory Structure (Expected)**

/murmur\_project  
  /backend\_server  
    \- main.py (FastAPI)  
    \- model.py (PyTorch Classes)  
    \- weights/murmur\_global.pth  
    \- user\_data/ (Stores prototypes)  
  /mobile\_app  
    \- lib/  
      \- main.dart  
      \- services/api\_service.dart  
      \- screens/recording\_screen.dart

