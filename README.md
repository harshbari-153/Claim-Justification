# Claim-Justification using Subject Predicate Distance

This repository contains the code and dataset for my MTech dissertation project on *Claim-Justification using Subject Predicate Distance*, an NLP model aimed at detecting inconsistencies between claims (single sentences) and supporting justifications (multiple sentences). The project introduces a word-embedding-based technique that significantly reduces processing time and computational requirements while maintaining accuracy levels comparable to sentence-BERT.

## Project Details

- **College**: National Institute of Technology, Surat - 395007, Gujarat
- **Course**: M.Tech CSE in Data Science
- **Batch**: 2023-2025
- **Guide**: Dr. Krupa Jariwala
- **Student**: Harsh Bari
- **Enrollment Number**: P23DS004

## Project Overview

Many real-world applications require validating claims based on provided justifications, such as in political speeches, advertisements, and resumes. Traditional methods like sentence-BERT, though accurate, are computationally intensive because they process entire sentences. This project proposes an optimized approach, focusing only on subject-predicate embeddings, to achieve faster and more efficient results.

### Key Features
- **Efficient Word Embeddings**: Processes only essential word embeddings (subject and predicate) in the claim and justification, reducing computational load.
- **Multi-Classifier Model**: Utilizes five binary classifiers to handle six ordinal classes, achieving accuracy on par with sentence-BERT.
- **Real-Life Applications**: Applicable to scenarios like identifying fake political claims, misleading advertisements, and resume inconsistencies.

## Folder Structure

This project is organized into nine main folders as follows:

1. **`1 extract justification`**: Contains `scrape_justification.py`, which scrapes over 21,000+ justifications using Beautiful Soup and saves them in the `justifications` folder as text files.

2. **`2_1 generate sentiments`**: Contains a Python script that extracts Natural Language Inference (NLI) sentiments from the justifications, resulting in 21-point feature vectors used for training in the subsequent models.

3. **`2_2 generating embeddings`**: Contains a Python script that extracts sentence embeddings from the justifications using sentence-BERT (768-dimensional embeddings), which serve as input features for model training as a comparison baseline.

4. **`2_3 generate distance`**: This folder contains the core research code of the project. The script extracts subject-predicate distance-based embeddings using Wikipedia word embeddings, resulting in 100-point embeddings used in training the model.

5. **`2_4 generate distance 2`**: Contains a similar process to `2_3 generate distance`, but instead of Wikipedia embeddings, it uses sentence-BERT, producing 384-point embeddings for comparison in the training model.

6. **`3_1 create and train model`**: Appends the sentiments (from `2_1 generate sentiments`) to the embeddings (from `2_2 generating embeddings`) and trains five binary neural networks for the six-class task. Model accuracies:
   - Classifier 1: 65.51%
   - Classifier 2: 73.75%
   - Classifier 3: 77.00%
   - Classifier 4: 51.32%
   - Classifier 5: 62.32%

7. **`3_2 create and train model`**: Appends the sentiments (from `2_1 generate sentiments`) to the embeddings (from `2_3 generate distance`) and trains five binary classifiers. Model accuracies:
   - Classifier 1: 66.28%
   - Classifier 2: 73.76%
   - Classifier 3: 77.01%
   - Classifier 4: 53.27%
   - Classifier 5: 62.12%

8. **`3_3 create and train model`**: Appends the sentiments (from `2_1 generate sentiments`) to the embeddings (from `2_4 generate distance 2`) and trains five binary classifiers. Model accuracies:
   - Classifier 1: 65.38%
   - Classifier 2: 73.71%
   - Classifier 3: 78.36%
   - Classifier 4: 53.69%
   - Classifier 5: 62.02%

9. **`dataset`**: Contains the project dataset in JSON format, used for various stages of processing and model training.

## Methodology

The project employs a six-class (ordinal) multi-classification approach by creating five binary classifiers. Each model achieved accuracy comparable to sentence-BERT but with **33% faster training time** and **50% less computational demand**.

### Dataset
- **Source**: Scraped 21,000+ articles using Beautiful Soup for NLP processing and model training.
- **Focus**: Includes examples from political speeches, advertisements, and sample resumes, allowing for a robust assessment of the claim-justification model.

## Results

Our approach achieved the same accuracy as sentence-BERT across all classifiers, with **33% less training time** and **half the computational resources**. This efficient method is suitable for deployment in real-time or resource-constrained environments. Specific accuracy results for each classifier model are provided in the folder descriptions above.

## Future Work

Potential extensions of this project include:
- Applying the model to additional NLP tasks, such as misinformation detection or sentiment analysis.
- Enhancing the classifier ensemble for broader class distinctions beyond ordinal classification.
- Further optimizing embedding techniques to explore hybrid approaches with sentence-based methods.

## Conclusion

This project provides an efficient, scalable solution for claim-justification matching in NLP, with practical applications in detecting inconsistencies across various domains. By focusing on essential components of claims and justifications, the model offers substantial improvements in processing speed and computational efficiency without sacrificing accuracy.

## Contribution

This project was developed as part of an MTech dissertation. Contributions included scraping and processing a large dataset, developing efficient embeddings based on subject-predicate distance, and implementing a multi-class classification model for claim-justification matching. The project also explores the potential for optimized NLP techniques in detecting inconsistencies in various real-world scenarios.
