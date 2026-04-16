# Instructor materials — course project overview

The text below is preserved from the course project handout in google docs, converted to markdown cuz im too lazy to keep switching from my code editor to the google doc

---

## Project overview

There is a quarter-long course project of building deep learning models for various application tasks. The main purpose of this project is to help students get familiar with classic deep learning models and use them to solve concrete real-world problems. The project will account for **40%** of your final grade, and it will have multiple stages:

- **Stage 1:** Group formation and programming environment setup. (0%)
- **Stage 2:** First trial with PyTorch and build up a Multi-Layer Perceptron model to classify the instances in a provided dataset. (10%)
- **Stage 3:** Object recognition from images with convolutional neural network model. (10%)
- **Stage 4:** Text classification and generation with recurrent neural network models. (10%)
- **Stage 5:** Network embedding and node classification with graph neural network models. (10%)

---

## Stage 1: Group formation and environment setup

*(Code template, report template)*

You need to find teammates to form a group with **four students** (more than four is not allowed). Create a cool name for your group. Submit a report to Canvas that includes your team name, student names, student IDs, and emails. Only one submission will be needed for each team, and the TA will help tie the group members up throughout the semester. You can either elect a team leader to do the submission, or you can take turns to take the responsibility for submissions for each stage. If one happens to leave/drop the class, the others should be on your own. So choose your groupmates carefully!

**Note 1:** If you prefer to work independently, then the solo mode should also be ok, but your solo project will be graded with the same criteria as others’ team project.

**Note 2:** Choose your team members carefully. (Some teams may end up with some unhappy issues or conflicts…)

**Note 3:** If you have difficulties forming a team, here are the solutions: (1) actively contact your peers in the same class; (2) we will do random team assignments if you really cannot find a team at the end (we cannot guarantee who your teammates will be :( ).

### Environment setup

1. **1-1:** Make sure your computer has the latest Python 3.x installed. If you don’t have Python, consider downloading it at [python.org/downloads](https://www.python.org/downloads/).

2. **1-2:** Download and install [PyCharm](https://www.jetbrains.com/pycharm/) and verify it can work correctly by creating a toy “Hello world” project in Python 3.x. (Please use the educational license, and don’t pay money to buy anything.)

3. **1-3:** You have two ways to install sklearn. (1) Install “scikit-learn” into your PyCharm project as shown by [this video](https://www.youtube.com/watch?v=f4OvzM7x5jo&ab_channel=AamirAlamgir), or (2) download and install sklearn following [scikit-learn installation](https://scikit-learn.org/stable/install.html).

4. **1-4:** You have two ways to install PyTorch. (1) Install “torch” into your PyCharm project similar to the above, or (2) follow [PyTorch get started locally](https://pytorch.org/get-started/locally/).

5. **1-5:** Verify PyTorch and sklearn both work correctly in PyCharm by creating toy projects that import these toolkits.

6. **1-6:** Download the provided code template. Set up the interpreter and other packages, change your current working directory to the `stage_1_script` folder, try to run the four scripts for stage 1. If PyTorch and sklearn are installed and the environment is set up correctly, you should be able to run the code and see the generated results.

7. **1-7:** Read through the code template carefully and understand how the code is organized; future project code will be based on this template.

8. **1-8 (optional):** For the MLP method, change the loss function and optimizer, as well as their setups (e.g., learning rates or other hyper-parameters) to see their impacts on model performance.

---

## Stage 2: Data classification with MLP (PyTorch)

*(Dataset, report template)*

This stage aims to help students get familiar with **PyTorch** and **scikit-learn (sklearn)**. Please write your first program with PyTorch to implement the MLP model introduced in class.

1. **2-1:** Download the dataset provided by the instructor (training and testing). Inspect the dataset before writing your code.

2. **2-2:** Write your own code based on the template for stage 2 (copy code from stage 1 into stage 2 and fix imports). You can reuse or change Stage 1 code (e.g., new `Dataset_Loader`, adapted `Setting`, adapted `Method_MLP`, more evaluation metrics). For multiclass classification, use weighted/macro/micro versions of F1, recall, and precision—not binary-only metrics.

3. **2-3:** Train the MLP on training data, apply to the test set, generate learning curves, and report final evaluation.

4. **2-4:** Try different architectures, loss functions, optimizers, and settings to improve performance.

5. **2-5:** Write a report using the provided template (**5 pages maximum**; longer reports may be penalized).

6. **2-6 (optional):** If you have a CUDA GPU, try running on GPU (e.g., [W&B: How to use GPU with PyTorch](https://wandb.ai/wandb/common-ml-errors/reports/How-To-Use-GPU-with-PyTorch---VmlldzozMzAxMDk)).

---

## Stage 3: Image classification with CNN

*(Dataset, report template)*

This stage introduces **convolutional neural networks (CNNs)** for object recognition on image data.

1. **3-1:** Download the three image datasets (hand-written digits, human faces, colored objects). For the face dataset: grayscale from color with equal R/G/B per pixel—three channels with equal values; one channel is often sufficient.

2. **3-2:** Write your own CNN in the provided template.

3. **3-3:** Train on digit images, test on digit test set, learning curves and metrics.

4. **3-4:** Train two more CNNs on faces and colored objects respectively; report learning curves and results.

5. **3-5:** Experiment with depth, kernel size, padding, stride, pooling, hidden dimensions, loss, etc.

6. **3-6:** Report (**5 pages maximum**).

7. **3-7 (optional):** CUDA GPU.

---

## Stage 4: Text classification and generation with RNN

*(Dataset, report template)*

This stage introduces **recurrent neural networks (RNNs)** for text classification and generation.

1. **4-1:** Download the two text datasets and inspect them.

2. **4-2:** Implement RNN models in the template for text classification.

3. **4-3:** Train for classification, evaluate on the test set, learning curves and metrics.

4. **4-4:** Train on the text generation dataset; generate a story from three starting words; compare to training data where appropriate.

5. **4-5:** Try LSTM and GRU and other architectural changes; repeat 4-2–4-4 and report.

6. **4-6:** Report (**5 pages maximum**).

7. **4-7 (optional):** CUDA GPU.

---

## Stage 5: Graph embedding and node classification with GNN

*(Dataset, `Dataset_Loader`, report template)*

This stage introduces **graph neural networks (GNNs)** for embedding and node classification.

1. **5-1:** Download the three graph datasets and inspect them.

2. **5-2:** Implement a **GCN** in the template for graph embedding and node classification.

3. **5-3:** Train on **Cora**, test, learning curves and metrics.

4. **5-4:** Repeat on **Pubmed** and **Citeseer**.

5. **5-5:** Report (**5 pages maximum**).

6. **5-6 (optional):** CUDA GPU.

---

## Useful links (from the original handout)

- Python: [python.org/downloads](https://www.python.org/downloads/)
- PyCharm: [jetbrains.com/pycharm](https://www.jetbrains.com/pycharm/)
- scikit-learn install: [scikit-learn.org/stable/install.html](https://scikit-learn.org/stable/install.html)
- PyTorch: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
- PyTorch Geometric (Stage 5): [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
