# Part 4 - Classification Writeup

After completing `a6_part4.py` answer the following questions

## Questions to answer

1. Comment out the StandardScaler and re-run your test. How accurate is the model? Why is that?
It is 62% accurate, this is because the numbers are non-standard so it always assumes they don't buy and only guesses 0.
2. How accurate is the model with the StandardScaler? Is this model accurate enough for the given use case? Explain.
0.69, this is not very accurate but for this given circumstance it is good enough to give a general guess where it can predict who is twice as likely to buy a car.
3. Looking at the predicted and actual results, how did the model do? Was there a pattern to the inputs that the model was incorrect about?
It was not very accurate at guessing, I could not find a pattern in the inconsitencies.
4. Would a 34 year old Female who makes 56000 a year buy an SUV according to the model? Remember to scale the data before running it through the model.
No they would not.
