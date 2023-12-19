
# Possible problems:

## 1-
***Discrim overfits to expert -> the generator do not get good feedback then? if this is the probelm ->***

### Possible Solution:

- [ ] 1 - maybe update discrim with a certain probability (random unif < prob of updating discrim) 

- [ ] 2 - lower the training itration of discirm after a while

- [ ] 3 - modifiy expert data in a way that is not complete in each updating iteration. in other words, not using the full expert data.

- [ ] 4 - **(too much maybe?!)** use different envs in each iteration


## 2- 

***The log_prob() to return `inf` and policy gets `nan` weights -> The policy is changing too much after each iteration of updating it? (data is off-policy)-> if this is the probelm ->***

### Possible Solution:

- [ ] 1 - We can lower the number of `num_gen_update`
- [ ] 2 - We can use bigger batch size `batch` so that the policy do not change too much and  the data stays valid (on-policy)
- [ ] 3 - We can lower `lr` to not change policy too much (but it is already too low)



## 3 -

***The end node is not seen -> does value net give good value for it? what about the discrim? if this is the probelm ->***


### Possible Solution:

- [ ] 1 - we have to pass the end node from expert to the discrim? but the learner should not have the end.
(possible that discrim gets bad as the learner does not have the end but expert has)

- [ ] 2 - have the end nodes in the both trajs.but mask the end campletely in calculate return 
(not passing the end goal to the policy network, and masking the value as we already do if new_states are end)



## 4 - 

***Too much repetitive data by passing Start and End data to the network -> may cause bad gardients or unstable learning? if this is the probelm ->***


### Possible Solution:
- [ ] Remove the start and the end from network (but what if we cut the traj `shorten_traj()` -> should we keep the start nodes and remove ends only?)



## Other things to try

- [ ] Maybe normalize the return?
- [ ] Forcing the weights of embeddings in discrim and value to be the same as policy 
