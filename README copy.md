# twitter-project
scrape and time series

With this project I am scraping twitter with the Daily_scrape jupyter notebook to get tweets the mention Amazon, Apple, CrowdSource, etc
And getting daily download of stock prices and doing rolling averages of prices and percentage changes and a naive beta calculation
Then building a sentiment classification neural network using the twitter_gru jupyter notebook
Then using this to classify the scraped tweets with the sentiment_from_nn notebook
After this classification, the duplicate tweets are removed using the post_sentiment_no_dupes notebook - 
     keeping copies of the last tweet for some files if that makes a difference, and the first tweets 
Then the no_dupes files are pre-processed for NN time series analysis with the time_series_preprocess jupyter notebook
I haven't completed the pre-processing yet (as of 6/26/21).
