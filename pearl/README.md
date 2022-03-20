# Reproduction of [PEARL](https://github.com/katerakelly/oyster) under [unstable_baselines](https://github.com/x35f/unstable_baselines) framework

# Quick Notes
    * A newer sac with automatic entropy tuning is implemented and tested. The performance matches the original implementation. Switch the sac version with the "use_new_sac" argument.
    * No raw z initialization at the "extra posterior" sampling step. Uses context from the train encoder buffer for z intialization before eollecting data