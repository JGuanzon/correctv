# Should we Correct for Peculiar Velocities?

**Project Details**

In supernova cosmology, peculiar velocities are often seen as just a source of noise.  As such, the effects of peculiar velocities are estimated from large scale structure, and then ‘correct’ by changing the observed redshift.  In this project, we’ll assess if these peculiar velocity corrections are a good idea.  Do they significantly improve our accuracy, or do they bias our precision?

**To Do List**

**18/07-22/07**
- [ ] Try to figure out what is going wrong with the code.
- [x] Optimise code + add counter/timer.
- [x] Run multiple times at different inputs.
- [ ] Test the probability functions.


**11/07-15/07**
- [x] Create hubble diagram of mod and redshift.
- [x] Generate fake data (including adding sigma=0.14 normal distribution to mod).
- [ ] Observe the effects on the hubble diagram.

**04/07-08/07**
- [x] Check 'mistake' on emcee (the missing 2pi). *Note: Running it with and without 2pi didn't change the outcome.*
- [x] Read [Davis T, et al](http://arxiv.org/abs/1012.2912) paper. *Note: Skipped over the complicated covariance matrix part.*
- [x] Read Ed's paper. *Note: Very briefly looked over it.*
- [x] Review background information for all the different variables.
- [x] Review [Ed's code](https://github.com/EdMacaulay/Spectroscopic_SN_min_ChSq).
- [ ] Modify [Ed's code](https://github.com/EdMacaulay/Spectroscopic_SN_min_ChSq) in terms of what we need.
- [x] Review physics behind distances (eg luminousity distance).
- [ ] Review about calculating density/mass maps to pecular velocity field correction to hubble diagram scattering.
- [ ] Generate Hubble's diagram with and without pecular velocities.
- [ ] Fit for the dark energy parameters omega_0 and omega_alpha.

**27/06-01/07**
- [x] Go through the [Fitting a Model to Data](http://dan.iel.fm/emcee/current/user/line/) by Friday.
- [x] Python + GitHub setup
- [x] Generate probablisitic model
- [x] Least squares fit + Linear algebra
- [x] Maximum likelihood estimation
- [x] Marginalization & uncertainty estimation
- [x] Results
- [x] Read more about relevant probability/statistics
- [x] Read more about peculiar velocities