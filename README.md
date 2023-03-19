# speed_dating

## Aims
- Can we define the perfect combination of features that equate to love?
- Using information from the speed dating dataset thats each person fills in **before** they meet their partner, can we predict if they will be a match?
- Which features most strong indicate whether a couple will match or not?

## Approach
- Pivot data so we get one row per date
- Apply data cleaning, check levels of missingness
- Try basic linear model, then implement feature engineerig to rty to improve performance.
- Try cluistering a decision trees- here I use a simple model in order to be able to extract feature importance and model insights
- Try more complicated models (lightgbm, NN) to see if performance can be improved

## Findings
- Not enough data to create a high performing model
- The similarity between how attractive someone wants their partner to be and how attractive the partner thinks they are has a great impact on a match
- The similarity between how intellegent someone wants their partner to be and how intellegent the partner thinks they are has a great impact on a match

## To run
- Run data_cleaning notebook and then run model notebooks

## TODO
- Try downsampling (although i suspect the data size is too small for this)
- Implement error analysis
- convert form notebooks to .py files
- add gitignore file