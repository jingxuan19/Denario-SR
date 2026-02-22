\\subsection{Exploration of Adaptive Learning Rates}

  \\item[4.1] Data Preparation and Initialization
    Begin by preparing the dataset for experimentation. Ensure that the data is normalized and any necessary preprocessing steps are taken to prepare the data for training. Initialize the model parameters using a uniform distribution.

  \\item[4.2] Mathematical Analysis of Adaptive Learning Rates
    Derive the mathematical formulation for adaptive learning rates based on adaptive gradient descent methods such as Adam or SGD. Analyze why adaptive learning rates exhibit smaller differences in performance compared to non-adaptive methods, which can be attributed to the ability of these methods to adapt to changing learning rates.

  \\item[4.3] Design and Implementation of New Optimizer
    Based on the observations from the mathematical analysis, design a new optimizer that is invariant to reparametrization. This can be achieved by incorporating adaptive learning rate mechanisms into the optimization algorithm. Implement the new optimizer using a suitable programming language (e.g., Python) and test its performance on the toy problem.

  \\item[4.4] Connection to Natural Gradient Descent
    Connect the new optimizer to Natural Gradient Descent (NGD) by analyzing how the adaptive learning rate mechanism affects the natural gradient computation. Investigate whether NGD remains invariant to reparametrization when using this new optimizer.

  \\subsection{Implementation of NGD for Toy Problem}

  \\item[5.1] Algorithm Implementation
    Implement NGD according to the provided notes, using a suitable programming language (e.g., Python). Ensure that the implementation is efficient and easy to understand.

  \\item[5.2] Testing and Evaluation
    Test the implemented NGD algorithm on the toy problem and evaluate its performance by comparing it with other optimization algorithms (e.g., SGD, Adam).

  \\subsection{Open Source Implementation of K-FAC}

  \\item[6.1] Choosing a Suitable Optimizer
    Select an open source implementation of an optimizer such as K-FAC that can be used for this study. Ensure that the chosen optimizer is suitable for the task at hand.

  \\item[6.2] Reasoning about Invariance
    Reason about whether K-FAC is invariant to the specific reparametrization and architecture used in this study. Analyze how the adaptive learning rate mechanism affects the optimization process.

  \\subsection{Proof of Invariance}

  \\item[7.1] Mathematical Derivation
    Derive the mathematical formulation for the invariance of NGD under smooth, differentiable reparametrisation. Use techniques from differential geometry and calculus to establish the proof.

  \\item[7.2] Proof Verification
    Verify the correctness of the proof by checking that it aligns with established results in optimization theory and differential equations.

  \\subsection{Discussion on Fisher Information}

  \\item[8.1] Data Collection
    Collect data on the performance of NGD under different reparametrizations and architectures. Ensure that the dataset is representative of the task at hand.

  \\item[8.2] Analysis of Merits
    Analyze the merits of using Fisher information averaged over test vs training data for optimization tasks. Discuss the advantages and disadvantages of this approach.