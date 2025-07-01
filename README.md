# ClaimLeaf 🌿  
**Your AI-Powered Crop Claim Companion**

ClaimLeaf is a lightweight, intelligent assistant that detects crop leaf diseases from images and instantly generates insurance-ready reports—streamlining claims for farmers, insurers, and agri-cooperatives.

## 🧠 Project Overview

Traditional crop insurance claims are slow, manual, and often inaccurate. ClaimLeaf addresses this by automating disease detection and claim reporting using deep learning and transfer learning.

---

## 🚀 Features

- 📷 Leaf disease detection via image upload  
- 📄 Instant PDF claim generation (offline-ready)  
- 🌐 Web/mobile interface with offline support  
- 🤖 ~93% classification accuracy, <1s inference  

---

## 🧱 Tech Stack

| Component        | Tool                             |
|------------------|----------------------------------|
| **Model**        | MobileNetV2 (pre-trained, fine-tuned) |
| **Data**         | PlantVillage (20K labeled images)|
| **UI**           | Streamlit + Image Picker         |
| **Report Generator** | FPDF (PDF output)            |
| **Helper**       | Gemini 2.5 fallback (<70% conf)  |

---

## 🔧 Model Training

Two-phase transfer learning:

1. **Feature Extraction** (Frozen base):  
   - LR: 0.001 | Epochs: 10 | Accuracy: 91%
2. **Fine-Tuning** (Unfrozen base):  
   - LR: 0.00001 | Epochs: 5 | Accuracy: 93%

- Input shape: `224x224 RGB`  
- Augmentation applied to reduce overfitting.

---

## 🧪 How It Works

1. Take or upload a leaf image  
2. AI model classifies disease (or detects healthy)  
3. Download a PDF report ready for insurance submission  

---

## 💼 Business Impact

| Stakeholder   | Impact                            |
|---------------|-----------------------------------|
| **Farmers**   | Faster payouts, no paperwork      |
| **Insurers**  | Lower fraud, reduced costs        |
| **NGOs/Govs** | Real-time disaster response       |

---

## 🛣️ Roadmap

- Expand to more crops (e.g., maize, wheat)  
- Integrate GPS and field data caching  
- Launch pilot with insurers/co-ops  
- Explore monetization (freemium/API/SaaS)

---

## 👨‍👩‍👧‍👦 Team

Lucas Ihnen · Massimo Tassinari · Omar Harradi  
Taddeo Carpinelli · Tomás Silva  
Prof. Concepción Díaz (Instructor)  
