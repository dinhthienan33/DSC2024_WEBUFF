# 🎭 Multimodal Sarcasm Detection on Vietnamese Social Media Texts 🚀

Welcome to the official repository for the **Multimodal Sarcasm Detection on Vietnamese Social Media Texts** project! This project tackles the fascinating and challenging task of detecting sarcasm in Vietnamese social media content, where sarcasm can manifest in **texts**, **images**, or **both**. Leveraging cutting-edge multimodal learning techniques, we aim to mimic the human brain's ability to process information from multiple data modalities effectively.

---

## 🌟 **1. Introduction**

### � **1.1 What is Multimodal Sarcasm Detection?**
Multimodal learning is an exciting field in machine learning that focuses on mimicking the human brain's ability to receive and process information from various data modalities, such as text, images, and audio. One of the most challenging tasks in this domain is **detecting sarcasm in social multimedia data**. Sarcasm can exist in **statuses**, **images**, or **comments** of a post, making it a complex problem to solve. 

In this task, participants are required to determine whether sarcasm is present:
- **Only in the image**
- **Only in the text**
- **In both the image and text**
- **Not present at all**

### � **1.2 Challenges of the Task**
The task presents several unique challenges:
- **Multimodal Complexity**: Sarcasm can be expressed through a combination of text and images, requiring models to understand the interplay between these modalities.
- **Language Specificity**: The dataset is in **Vietnamese**, adding complexity due to linguistic nuances and cultural context.
- **Data Imbalance**: The dataset may be imbalanced, with sarcastic examples being rarer than non-sarcastic ones.
- **Contextual Understanding**: Sarcasm often relies on context, making it difficult for models to detect without a deep understanding of the content.

### 📊 **1.3 About the Data**
The dataset consists of **Vietnamese social media posts** collected from platforms like Facebook, Instagram, and Zalo. Key highlights:
- **Modalities**: Text (statuses, comments) and images.
- **Size**: [e.g., 10,000 posts with text and image pairs].
- **Labels**: Each post is labeled as:
  - Sarcasm in text only
  - Sarcasm in image only
  - Sarcasm in both text and image
  - No sarcasm
- **Preprocessing**: [Describe any preprocessing steps, e.g., text cleaning, image resizing, etc.]

---

## 🧠 **2. Method Overview**

Our solution achieved an impressive **rank of 10/43** on the private test set! 🎉 Here's a breakdown of our approach:

### 🛠️ **4.1 VisoBert + Beit Training**
- **Model Architecture**: We combined **VisoBert** (for text understanding) and **Beit** (for image understanding) to create a robust multimodal model capable of detecting sarcasm in both text and images.
- **Handling Imbalanced Data**: To address data imbalance, we employed techniques such as **oversampling**, **weighted loss functions**, and **data augmentation**.
- **Parameters**: Here are the key parameters used:
  ```python
    {'seed_val': 0,
    'training_size': 9724,
    'dev_size': 1081,
    'test_size': 1413,
    'num_train_epochs': 10,
    'batch_size': 32,
    'learning_rate': 1e-05,
    'weight_decay': 0.01,
    'warmup_steps': 0,
    'max_seq_length': 512}


### 🎨 **4.2 CLIP (Multimodal Model) Training**

- **Model Architecture**: 
  - Utilizes the **CLIP** (Contrastive Language–Image Pretraining) model, which combines a **vision transformer (ViT)** for image encoding and a **text transformer** for text encoding.
  - A custom **MultimodalClassifier** is added on top of CLIP to concatenate image and text features and predict sarcasm labels (e.g., sarcasm in text, image, both, or none).
  - The model is fine-tuned end-to-end, allowing both the CLIP backbone and the classifier to adapt to the Vietnamese sarcasm detection task.

- **Training Details**:
  - **Training Parameters**:
    - **Epochs**: 30
    - **Batch Size**: 256
    - **Learning Rate**: 1e-2
    - **Weight Decay**: 0.001 (for regularization)
    - **Warmup Steps**: 2000 (for learning rate scheduling)
  - **Optimizer**: AdamW with cosine learning rate scheduling.
  - **Loss Function**: CrossEntropyFocalLoss with class weights to handle imbalanced data.
  - **Mixed Precision Training**: Enabled using `torch.amp` for faster training and reduced memory usage.
  - **Gradient Clipping**: Applied with a max norm of 1.0 to prevent exploding gradients.
  - **Device**: Training is performed on GPU (if available) for efficient computation.

- **Key Highlights**:
  - **Fine-Tuning**: The entire CLIP model is fine-tuned, including both the vision and text transformers, to adapt to the specific nuances of Vietnamese social media data.
  - **Imbalanced Data Handling**: Class weights are used in the loss function to address the imbalance between sarcasm and non-sarcasm examples.
  - **Scalability**: The architecture is designed to handle large-scale datasets and can be extended to other multimodal tasks.

- **Training Workflow**:
  1. **Data Preparation**:
     - The dataset is loaded and preprocessed, with images and texts tokenized using CLIP's tokenizer.
     - A custom `MultimodalDataset` and `DataLoader` are used to handle batching and oversampling for imbalanced classes.
  2. **Model Training**:
     - The model is trained for 30 epochs, with each epoch iterating over the entire dataset.
     - Mixed precision training and gradient clipping are applied to optimize training efficiency and stability.
  3. **Evaluation**:
     - After each epoch, the average loss and learning rate are logged to monitor training progress.
  4. **Model Saving**:
     - The model's state is saved after training for future inference or further fine-tuning.

- **Performance**:
  - The model achieves competitive performance on the Vietnamese sarcasm detection task, leveraging CLIP's multimodal capabilities and fine-tuning on the specific dataset.

This approach ensures robust and efficient training for multimodal sarcasm detection, making it suitable for real-world applications on Vietnamese social media platforms.

### 🤝 **4.3 Ensemble Method (Voting)**

-   **Ensemble Strategy**: To further enhance performance, we combined predictions from **VisoBert+Beit** and **CLIP** using a **voting mechanism**. This approach allowed us to make more robust and accurate decisions by leveraging the strengths of both models.

-   **Diagram Instruction**: Here's how to visualize the ensemble method:

    1.  Draw two boxes representing **VisoBert+Beit** and **CLIP**.

    2.  Connect both boxes to a third box labeled **"Ensemble Voting"**.

    3.  Add arrows from the ensemble box to the final output.

* * * * *

📈 **3\. Results**
------------------

Here's a glimpse of our achievements:

-   **Rank**: 10/43 on the private test set.

-   **Performance Metrics**: [Include specific metrics, e.g., accuracy, F1-score, etc.]

![Proof of Results](assets\score.png)\
*Proof of our rank and performance metrics.*

* * * * *

🌱 **4\. Contribution**
-----------------------

Our work contributes to the field by:

-   Introducing a novel combination of **VisoBert**, **Beit**, and **CLIP** for **multimodal sarcasm detection** in Vietnamese social media data.

-   Demonstrating the effectiveness of ensemble methods in handling **multimodal and imbalanced data**.

-   Providing a scalable and reproducible solution for **sarcasm detection in low-resource languages** like Vietnamese.

* * * * *

📜 **5\. License**
------------------

This project is licensed under the **[License Name, e.g., MIT License]**. Feel free to use, modify, and distribute the code as per the license terms.

* * * * *

🗺️ **Outline**
---------------

1.  **Introduction**

    -   1.1 What is Multimodal Sarcasm Detection?

    -   1.2 Challenges of the Task

    -   1.3 About the Data

2.  **Method Overview**

    -   4.1 VisoBert + Beit Training

    -   4.2 CLIP (Multimodal Model) Training

    -   4.3 Ensemble Method (Voting)

* * * * *

For questions, collaborations, or just to say hi, feel free to reach out at dithienan03@gmail.com. Let's build the future of AI together! 🚀