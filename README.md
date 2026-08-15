# NeuroScan AI: Adversarially Robust Brain Tumor Detection

NeuroScan AI is a full-stack MedTech SaaS platform designed to provide secure, clinical-grade brain tumor segmentation. Utilizing the custom ARMT-GAN (Adversarially Robust Medical Tumor Segmentation) framework, the platform ingests both 2D MRI slices and 3D NIfTI volumes to generate highly accurate tumor masks and Explainable AI (XAI) heatmaps.

## 🚀 Key Features
* **ARMT-GAN Inference Pipeline:** Achieves high spatial precision and adversarial robustness against medical imaging artifacts.
* **XAI Heatmap Verification:** Generates color-coded activation maps to provide clinical interpretability for AI-driven diagnoses.
* **3D NIfTI & 2D Image Support:** Seamlessly processes raw Kaggle datasets (.jpg) and complex BraTS datasets (.nii/.gz).
* **Automated Clinical Reporting:** Compiles diagnostic visuals and patient metadata into downloadable PDF reports.
* **Asynchronous Processing:** Utilizes Celery and Redis to handle heavy tensor operations without blocking the API layer.
* **Medical-Grade Security:** Hard-locked Firebase authentication ensuring only authorized institutional personnel can access the terminal.

## 🛠️ Technology Stack
* **Frontend:** Next.js, Tailwind CSS, Lucide React (Glassmorphic, interactive HTML5 canvas UI).
* **Backend:** FastAPI, Python, SQLAlchemy, PostgreSQL.
* **AI/ML:** PyTorch, TorchVision, Nibabel (for neuroimaging).
* **Infrastructure:** Celery (Task Queue), Redis (Broker), MinIO (S3-compatible Object Storage), Firebase (Auth).

## 👨‍💻 Author
**Gagan N** 
* GitHub: [@Gagann-09](https://github.com/Gagann-09)
* Developed as an advanced end-to-end medical AI research project.