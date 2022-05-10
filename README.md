# chrome-review
# Question: 1

## Write a regex to extract all the numbers with orange color background from the below text in italics.

{"orders":[{"id":1},{"id":2},{"id":3},{"id":4},{"id":5},{"id":6},{"id":7},{"id":8},{"id":9},{"id":10},{"id":11},{"id":648},{"id":649},{"id":650},{"id":651},{"id":652},{"id":653}],"errors":[{"code":3,"message":"[PHP Warning #2] count(): Parameter must be an array or an object that implements Countable (153)"}]}

## For answe please refer regex.py

# Question: 2.
- Problem statement - There are times when a user writes Good, Nice App or any other positive text in the review and gives 1-star rating. 
- Your goal is to identify the reviews where the semantics of review text does not match rating. Your goal is to identify such ratings where review text is good, but rating is negativeso that the support team can point this to users.
- Deploy it using - Flask/Streamlit etc and share the live link.

# Ans:
I have shared the code here in GitHub. The datasets used is the same which was provided. The overall idea is as follows.
1. Read data from csv using pandas
2. Extract useful columns
3. Convert rating to a positivity score - which denotes the class to which it belongs to. Rating of above 3 is classified as 1, other ratings as 0.
4. i used TfidfVectorizer. TfidfVectorizer uses an in-memory vocabulary (a python dict) to map the most frequent words to features indices and hence compute a word occurrence frequency (sparse) matrix.
5. Then, we train the model using support vector classifier we do not want to overfit the model as there is a chance that a good review can have a bad rating. Finally we validate using test data, but wouldn't go back to fine tune the model, as we are expecting erroneous classification from the original data in small proportions.
6. We are looking for those rows where the semantic analysis gives rating 1, while the review rating positivity is 0 - extract these rows and display
7. Put all of this into streamlit and accept inputs from a csv as test data

### Regarding the output: 
1. we have found that there are many entries in the output where there are bad reviews accompanied by bad ratings. This shouldn't have been counted however. The reason for this observation was that the training data was limited and as a result there was some limitations for ex, on max_features for vectorizer.

2. On the positive side, there are many bad ratings with good reviews also present in the output.
Hence, in conclusion, we need to do a better finetuning of the model with a larger dataset, possibly a known cleaner dataset and then use this model to predict such good review/bad ratings entries.

(please refer solution.py and stremlit.py)

3. The streamlit live link for deployed app is [link](https://share.streamlit.io/this-swapnil/chrome-review/main/streamlit.py)


# Quetion 3. Ranking Data - Understanding the co-relation between keyword rankings
with description or any other attribute.
## Ans: 
```
- Ranking has direct correlation to the keyword used in search and presence of that keyword in either the app_id directly or in the app description.

- Early presence of keyword will impact the ranking as even with humans we see that people tend to look for catch words in the initial couple of sentences.

- APP ID has direct impact on the ranking as the search keyword, if present in the app_id itself, will impact in improving the ranking of the app in playstore.

- Another parameter that would affect the ranking is how many times an user who is looking out for a particular functionality chooses to press the app link. That is determined by the type of catchy adjectives used to explain about the in the short description. Like easy to use, free to use, etc.

- Short description will be more catchy if they are precise, than long descriptions
```

# Part 2
```
Q. Check if the sentence is Grammatically correct: Please use any pre-trained model or use text from open datasets. Once done, please evaluate the English Grammar in the text column of the below dataset.
```
## Ans.
I have used a pre-trained language tool for performing the grammar check and it also provides the correct sentence suggestions. There are other libraries also similar to language-tool. It is the newest and has more stars and continuous contributors on GitHub.

language_check:

One such library is language_check. The language_check library doesn’t come bundled with Python. Instead, you have to manually download it from the command line or download it from pypi.org and then install it manually. The language_check specifies the mistakes along with Rule Id, Message, Suggestion, and line number in the document. Also using this we can directly correct the mistakes in the file only. It points out the mistakes accurately but it doesn’t guarantee 100% success in finding mistakes. Sometimes it may miss out on some important mistake. So relying on it completely is not advisable.
```
pip install --upgrade language-check
```
## Sample Output is:
```
Hye this app is very amazing  -- Mistakes found,  1  mistakes
This Apps needs some more Development!  -- Mistakes found,  1  mistakes
miceapp  -- Mistakes found,  1  mistakes
App crashed when I am on voice call please fix bugs  -- No mistakes
stupid app. never use again.  -- Mistakes found,  2  mistakes
Very wonderful app. I love it  -- No mistakes
```
Also, I created the <b>Correction.csv </b>file in which, if there is any mistake, the correct sentence is provided and also count the no of mistakes.

(please refer grammar_check.py and correction.csv)

# Bonus Points

# Question: Write about any difficult problem that you solved. (According to us difficult - is something which 90% of people would have only 10% probability in getting a similarly good solution).
## Ans.
    I haven't solved any difficult problem according to you something

# Question: Formally, a vector space V' is a subspace of a vector space V if
- V' is a vector space
- every element of V′ is also an element of V.

Note that ordered pairs of real numbers (a,b) a,b∈R form a vector space V. Which of
the following is a subspace of V?

- The set of pairs (a, a + 1) for all real a
- The set of pairs (a, b) for all real a ≥ b
- The set of pairs (a, 2a) for all real a
- The set of pairs (a, b) for all non-negative real a,b

## Ans: For a subspace H to be a vector space by itself, it has to fulfil the following 3 criteria.
1. Zero - The zero vector of vector space V must also be in H
2. Addition - For each u,v in H, u+v is also in H.
3. Scalar multiplication - For each u in H and a scalar c, cu is also in H.

Let us consider each of the above sets and discuss whether they form a subspace of V

1. The set of pairs (a, a + 1) for all real a

    >clearly (0,0) which is the zero of vector space V is not in this set. So it can’t be a subspace

2. The set of pairs (a, b) for all real a ≥ b
    > It meets the first two criteria discussed above. It has the zero vector (0,0) in it. It also satisfies the addition rule. But it fails to meet the scalar multiplication rule. Ex: let c = -5, u = (4,3) cu = (-20,-15) which is not an element of the given set.

3. The set of pairs (a, 2a) for all real a

    > It meets all the 3 criteria. It has the zero vector (0,0) in it. It meets the addition rule. Ex: for any u = (a,2a) and v = (b,2b), u+v = (a+b, 2(a+b)) which belongs to the given set. It also meets the scalar multiplication criterion. For any u = (a,2a), cu = (ca, 2ca) which belongs to the given set. So, the set of pairs (a,2a) for all real a, form the subspace of vector space V.

4. The set of pairs (a, b) for all non-negative real a,b
    > It satisfies the first two criteria for being a subspace as it contains both zero vector and satisfies the addition rule for all the elements in the set. But it fails to meet the scalar multiplication criterion, for a choice of negative value for scalar c. ex: if c = -1, an element of the set u = (3,4) , cu = (-3,-4) which is not a member of the given set.