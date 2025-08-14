# CarGrade – Car Evaluation (UCI)

**Description:** A predictive model that classifies a car's quality level (`unacc`, `acc`, `good`, `vgood`) based on features such as `buying`, `maint`, `doors`, `persons`, `lug_boot`, and `safety`.

---

## 📦 Tech Stack

* Python 3.10+
* scikit-learn, pandas, matplotlib
* Streamlit (optional web app)
* pickle (for saving models/encoders)

---

## 🗂️ Dataset

* **Source:** UCI ML Repository — *Car Evaluation*
* **Features:** Categorical
* **Target:** `class` (4 categories)
* Loaded automatically via URL in the code.

---

## 🖥️ Streamlit App

```bash
# Train the model first
streamlit run app.py
```