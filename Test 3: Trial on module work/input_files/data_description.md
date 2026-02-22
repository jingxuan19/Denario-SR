
Different parametrizations can behave differently under gradient descent and may lead to qualitatively different solutions. In this mini-project, you will look at how we can further understand, mitigate or remove the influence of parametrization.

This mini-project option is **more theory-focussed**, and asks you to experiment with Natural Gradient Descent (NGD) and related methods. We provide you with [some notes on NGD](https://hackmd.io/@fhuszar/H1-t95X3T) and some resources linked from there that explain how it is derived.

### Questions to Explore

You should not do all of these (you will notice they add up to more than 40 marks), these are just given as indication of the number of marks awarded for certain amount of work. You should 'build your own miniproject'.

* Explore Adaptive Learning Rates *(~10 marks)*
  * In Question A.5 of Assignment 1, you explored how using different optimizers affects training from equivalent initialisations. If you used Adam, you should have observed a smaller difference effect than when using SGD. *Discuss/illustrate mathematically why this happens. Can you design and implement a new optimiser based on your observations that is uneffected by the reparametrisation in A.5.2? Can you connect this to NGD?
*  Implement NGD for the toy problem in A.5 *(~30 marks for thorough solution)*
  * Follow the outline of the algorithm given in the notes, implement NGD. Use your algorithm on the toy task from Question A.5 and show that NGD displays invariance to reparametrisation. A couple notes:
    * There are multiple ways to compute the Fisher information matrix or the natural gradient. Your implementation does **not** need to be super efficient, but if you choose a very inefficient version, perhaps comment on how it could be made more efficient, or what the main bottlenecks are.
    * It's fine if you implement NGD specifically for this shallow MLP architecture, and if it does not work generally for arbitrary architectures. That said, we do encourage you to think about using generic automatic differentiation tools.
* Try an [open source implementation](https://github.com/n-gao/pytorch-kfac) of an optimizer like K-FAC. Reason about whether K-FAC is invariant to this specific kind of reparametrization in this specific architecture. *(~15 marks for a thorough exploration)*
* Prove the invariance of Natural Gradient Flow under smooth, differentiable reparametrisations *(~10 marks)*
* Discuss or illustrate the merits of using Fisher information averaged over test vs training data. *(max ~10 marks)*