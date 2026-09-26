# 🚀 DEPLOYMENT GUIDE: GET YOUR LIVE PROJECT URL FOR SIH PPT
# Project: SAMUDRA-AI (SIH 2026 Problem Statement 26057)

This project is built using **FastAPI + ONNX Runtime + Modern Navy Web Cockpit (HTML/CSS/JS + Leaflet GIS)**.
It has NO heavy dependencies, requires NO paid GPU, runs in under ~50MB RAM, and can be deployed for **100% FREE** with a permanent public URL that you can put directly in your SIH PPT!

---

## 🌟 OPTION 1: Deploy on Render.com (Recommended — Simplest 2-Minute Setup)

Render gives you a clean, permanent public URL like:
👉 `https://samudra-sonar-ai.onrender.com`

### Step-by-Step Instructions:
1. **Push your code to GitHub:**
   - Create a new repository on [GitHub](https://github.com/new), e.g., `samudra-sonar-ai`.
   - In this folder (`d:\CODE JAANI CODE\hackathorns\sih ka project 2`), open terminal and run:
     ```bash
     git init
     git add .
     git commit -m "feat: initial release of SAMUDRA-AI for SIH 26057"
     git branch -M main
     git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/samudra-sonar-ai.git
     git push -u origin main
     ```
2. **Go to [Render.com](https://render.com):**
   - Sign up / Log in with your GitHub account.
   - Click **"New +"** -> Select **"Web Service"**.
   - Select your `samudra-sonar-ai` repository.
3. **Configure Settings:**
   - **Name:** `samudra-sonar-ai`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** `Free`
4. **Click "Create Web Service":**
   - In 2 minutes, Render will build and deploy the application.
   - You will get your live link: `https://samudra-sonar-ai.onrender.com`!

---

## 🤗 OPTION 2: Deploy on Hugging Face Spaces (Best for AI Hackathons)

SIH and NIOT evaluators love Hugging Face Spaces because it is the global standard for ML prototypes.
URL Format:
👉 `https://huggingface.co/spaces/<YOUR_USERNAME>/samudra-sonar-ai`

### Step-by-Step Instructions:
1. Go to [Hugging Face](https://huggingface.co) and sign in.
2. Click **"New Space"** (`https://huggingface.co/new-space`).
3. Set:
   - **Space Name:** `samudra-sonar-ai`
   - **Space SDK:** Choose **Docker** (Blank).
   - **Visibility:** Public.
4. Clone the space repo locally or upload the project files (including the `Dockerfile`, `main.py`, `models/`, `engine/`, `static/`, and `data/` folders).
5. Hugging Face will automatically build the Docker container and make your app live!

---

## 💻 How to Test Locally on Your Machine (Right Now!)

To run and view the cockpit on your own computer:
1. Open terminal in this folder and run:
   ```bash
   python main.py
   ```
2. Open your browser and go to:
   ```
   http://localhost:8000
   ```
3. Test the features:
   - Click any target sample: **Ghost Net**, **Shipwreck**, **Submarine Pipeline**, or **Mine Cylinder**.
   - Watch the live **Acoustic Waterfall Scanner** overlay the bounding box.
   - See the **System 1 Laya Reflex Autopilot** react in under ~5ms.
   - Watch the **Leaflet Marine Map** pinpoint coordinates in the **Gulf of Mannar**.
   - Click **"COAST GUARD DISPATCH (PDF)"** to download the official mission order!
