# 📦 Smart Trash Bin — IoT & AI Waste Classification

<div align="center">
  <img src="https://github.com/user-attachments/assets/99603e1f-7c1a-4eb6-9a55-482555072f29" alt="SIC1" width="600"/>
</div>

---

## 📖 Project Overview

Smart Trash Bin is an **IoT-based prototype** that automatically classifies and sorts waste into categories: **Organic, Plastic, and Other**.  
The project integrates **ESP32-CAM, ultrasonic sensor, and servo motor** with an **image classification model**, and is deployed via **Streamlit** for real-time monitoring.

This project was developed as part of **Samsung Innovation Campus Batch 6 (2024/2025)**, where it successfully advanced to **Stage 4**.

---

## 🎯 Objectives

- Create an **affordable and scalable** waste management solution.  
- Support **Sustainable Development Goals (SDG 11 & 12)** for sustainable cities and responsible consumption & production.  
- Demonstrate the integration of **IoT devices, AI image classification, and web deployment** into a functional prototype.  

---

## ⚙️ Features

- 🗑️ **Automatic waste classification** (Organic, Plastic, Other)  
- 🎥 **ESP32-CAM integration** for image-based detection  
- 📡 **Ultrasonic sensor** to measure fill level  
- ⚙️ **Servo motor** to sort trash into corresponding bins  
- 🌐 **Streamlit dashboard** with:
  - **Overview** (classification summary)  
  - **Time Analysis** (waste detection trends)  
  - **Latest Detection** (real-time detection results)  

---

## 🛠️ Tech Stack

- **Hardware:** ESP32-CAM, Ultrasonic Sensor, Servo Motor  
- **Software & Frameworks:**  
  - Python  
  - Flask (backend API)  
  - Streamlit (web deployment & dashboard)  
  - OpenCV / Machine Learning (image classification)  
  - TensorFlow / Keras (MobileNetV2 for image classification)  
- **Database:** MySQL (optional for logging)  

---

## 🧠 Model Training

The image classification model was trained using **MobileNetV2** on a balanced dataset of 3 classes:
- Organic
- Plastic
- Other

Steps included:
- Data augmentation & balancing (15k images per class)  
- Transfer learning with MobileNetV2 (pretrained on ImageNet)  
- Achieved high accuracy on validation data  

---

## 🚀 Deployment

The application is deployed via **Streamlit**.  
To run locally:

```bash
# Clone this repository
git clone https://github.com/Zaps36/smart_trash_can-app.git
cd smart_trash_can-app

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
````

---

## 📸 Screenshots

* **Visualization Page** <img src="https://github.com/user-attachments/assets/96f38d60-54e5-4058-8345-5f053015c8a1" alt="Visualization" width="600"/>

* **Dashboard Overview** <img src="https://github.com/user-attachments/assets/0153d8d9-e832-4760-93f7-e73f8d4972bb" alt="Dashboard" width="600"/>

* **Prototype (Exterior & Interior)** <img src="https://github.com/user-attachments/assets/2a535ebc-c72d-46ab-bfdc-b9e193b30b27" alt="Exterior" width="400"/> <img src="https://github.com/user-attachments/assets/2bd433a0-20c1-4c0d-9b60-3340ff954609" alt="Interior" width="400"/>

---

## 🏅 Recognition

* **Samsung Innovation Campus (SIC) Batch 6, 2024/2025**
* Advanced to **Stage 4** with this project
* Certificate included in project documentation

---

## 👨‍💻 Contributors

* Farrel Laurensius Suryadi
* Jason Therawan
* Valentinus Ayodya Koesyudawisama
* Jerry Sebastian

---

## 📬 Contact

For inquiries, please reach out via:

* LinkedIn: [Jerry Sebastian](https://www.linkedin.com/in/jerrysebastian1/)
* Portfolio: [jerry-portofolio-six.vercel.app](https://jerry-portofolio-six.vercel.app/)

```


