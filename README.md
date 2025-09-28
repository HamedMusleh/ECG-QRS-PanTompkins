# 🫀 ECG QRS Detection using Pan-Tompkins Algorithm

## 📖 Overview
This project focuses on the **reproduction and enhancement of the Pan-Tompkins algorithm** for real-time **QRS complex detection** in ECG signals.  
The algorithm is implemented step-by-step to illustrate the signal processing pipeline and ensure accurate heartbeat detection.

---

## ⚙️ Methodology

### 🔍 Algorithm Stages
1. **Bandpass Filtering**  
   Removes baseline wander and high-frequency noise.  

2. **Derivative Operation**  
   Highlights the slope of the QRS complex.  

3. **Squaring Function**  
   Amplifies larger values and makes all data positive.  

4. **Moving Window Integration**  
   Smooths the signal and emphasizes QRS durations.  

5. **Adaptive Thresholding**  
   Uses moving average to dynamically detect QRS peaks.  

---

## 🧪 Experiments
- **Dataset:** MIT-BIH Arrhythmia Database.  
- **Validation:** The implementation was tested on real ECG signals.  
- **Results:**  
  - Accurate identification of QRS peaks.  
  - High sensitivity and specificity for normal signals.  

---

## 📊 Results
- The algorithm correctly identified QRS complexes across multiple test ECG signals.  
- Demonstrated robustness against baseline drift and moderate noise.  
- Visualized full **signal flow diagram** for clarity.  

---

## 🚀 Future Work
- Apply **LMS-based adaptive filtering** to improve robustness under high-noise conditions.  
- Explore integration with **wearable real-time ECG monitoring systems**.  

---

## 📂 Project Structure
