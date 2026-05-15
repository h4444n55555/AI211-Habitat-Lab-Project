# Vision-Language Reasoning for Visual Navigation Using Habitat Simulator

![Habitat Challenge](results/habitat_challenge_title.png)

## Overview
This repository contains the end-to-end implementation of a learning-based **Vision-Language Navigation (VLN)** system built on top of the photorealistic **Habitat Simulator** and **Matterport3D (MP3D)** environments.

The project explores how an autonomous agent can jointly understand egocentric visual perception and natural language instructions to predict discrete motor actions (move forward, turn left, turn right, stop). We implemented a modular multimodal architecture using pretrained **CLIP (ViT-B/16)** encoders fused via cross-attention, enabling generalized semantic reasoning for indoor navigation.

## Key Contributions
1. **Multimodal Cross-Attention Policy:** Built a robust baseline using frozen CLIP backbones and cross-attention fusion.
2. **Generalization & Ablation:** Quantified encoder contributions, demonstrating strong zero-shot resistance to paraphrased instructions (61.2%) and unseen environments (66.6%).
3. **Lightweight Edge Extension:** Engineered a compressed fusion variant that reduces trainable parameters by **71.7%** with less than a 0.5% drop in accuracy.
4. **Unified Navigation Paradigm:** Implemented specialized **ObjectNav** and **PointNav** policies, chained together into a continuous, multi-phase sequence (Find object -> Pick up -> Carry to coordinate).
5. **Interactive Navigation Demo:** Built a real-time OpenCV GUI tool to manually interact with and debug the agent inside the 3D simulator.

## Project Structure & Tasks

- **Task 1: Environment Setup:** Integration with Habitat Simulator and Matterport3D datasets. Implemented discrete kinematic transitions (0.25m step, 10° turn).
- **Task 2: VLN Baseline Architecture:** Formulated the CLIP cross-attention model. Achieved **70.32% validation accuracy** via Behavioral Cloning on shortest-path expert trajectories.
- **Task 3: GPU Scale-up & Evaluation:** Scaled training to 500 episodes on an RTX 3050 GPU. Identified the "Stopping Problem" — achieving an Oracle Success Rate of 60% but struggling with terminal `stop` prediction.
- **Task 4: Generalization:** Tested data scaling limits and evaluated performance on out-of-distribution language constraints.
- **Task 5: Lightweight Controlled Extension:** Shrunk the cross-attention dimensions to optimize for mobile/edge robotics.
- **Task 6: Object & Point Navigation:** Stripped out language for pure geometric PointNav (79% accuracy) and integrated categorical semantic ObjectNav. 

## Model Architecture

The core agent utilizes a **CLIP-based Dual-Encoder** setup:
1. **Visual Stream:** Egocentric RGB image $\rightarrow$ CLIP Vision Encoder $\rightarrow$ 197 Visual Tokens (768-dim $\rightarrow$ 512-dim).
2. **Text Stream:** Natural Language Instruction $\rightarrow$ CLIP Text Encoder $\rightarrow$ Text Tokens (512-dim).
3. **Fusion:** Transformer Cross-Attention layers where Text acts as the Query and Vision acts as the Key/Value.
4. **Action Head:** Masked mean pooling followed by an MLP mapping to a 4-dimensional categorical action distribution.

## Results Summary

| Experiment / Config | Best Val Accuracy | Key Insight |
| :--- | :--- | :--- |
| **Task 2 Baseline** | 70.32% | Stable convergence via mixed precision and label smoothing. |
| **Task 3 GPU Best (lr=2e-4, bs=8)** | 66.91% | Smaller batch sizes aided gradient stability. OSR hit 60%. |
| **Task 4 Paraphrased** | 61.24% | Strong robustness to natural language variation. |
| **Task 4 Frozen Text Encoder** | 17.52% | Fine-tuning text representations is critical for navigation. |
| **Task 5 Lightweight Fusion** | 67.64% | **-71.7% parameters** with negligible performance drop. |
| **PointNav** | 79.49% | Continuous geometric goals are simpler to optimize than semantics. |

## Interactive Demonstration

To qualitatively evaluate the policies, we developed a real-time interactive GUI. 

To launch the interactive session:
```bash
python scripts/interactive_nav.py --scene /path/to/mp3d/scene.glb
```

**Controls:**
- **Right Panel:** Click an object category to reveal instance markers on the map.
- **Top-Down Map:** Click anywhere to trigger the **PointNav** policy to navigate the agent to that coordinate.
- **Instance Markers:** Click a numbered marker to trigger the **ObjectNav** policy.
- `R` - Reset episode
- `Q` - Quit and save video recording

## Team / Contributors
* **Hans (2024AIB1011)**
* **Ravikant (2024AIB1013)**
* **Rishav (2024AIB1014)**

*Indian Institute of Technology Ropar* | *Machine Learning Project 2026*
