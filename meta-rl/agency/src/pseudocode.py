
# Case 1
# maksing both value and policy loss for observational trials
# currently implemented when masking = True

# Case 2 (Marcels case)
# value loss to 0 -> set value coef to 0
# policy loss uses everything (still maksing rewards for the advantage terms)

# Case 3
# not value_loss[self.mask] = 0 not policy loss to 0
# knows that its an observational trial
# still gets reward but cant use it for learning