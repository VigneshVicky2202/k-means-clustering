from sklearn.cluster import KMeans
import numpy as np

class DecisionMaker:
    def __init__(self):
        self.conditions = {
            "weather": {
                "rains": "It is raining.",
                "rainy": "It is raining.",
                "rain": "It is raining.",
                "getting more water": "It is raining.",
                "I drenched": "It is raining.",
                "shining sun": "It is sunny.",
                "sunny": "It is sunny.",
                "summer": "It is sunny and hot.",
                "hot": "It is hot.",
                "warm": "It is warm.",
                "scorching": "It is scorching hot.",
                "heatwave": "It is a heatwave."
            },
            "coffee_temperature": {
                "hot": "The coffee is hot.",
                "hotter": "The coffee is hot.",
                "chill": "The coffee is cold like thick water.",
                "warm": "The coffee is warm.",
                "cold": "The coffee is cold.",
                "iced": "The coffee is iced."
            },
            "heart_beats": {
                "Lup-tup": "Your heart beats are normal.",
                "faster like a bullet train": "Your are going to close your eyes at anytime. So, be careful",
                "nothing hears": "Deii you've already died. It's been an hour",
                "slow": "Soon you needs a treatment",
                "the pulse is dropping": "Get the glancer. Seek medical attention!"
            },
            "job_status_prediction": {
                "employed": "You have a job, Keep up the good work!",
                "i don't have a single penny": "Then you're useless in this world. That's what the rich people says that leads to you're jobless",
                "jobless": "You are currently jobless. Keep searching, and you'll find opportunities!",
                "balamurali & co": "You're the smart people now. you're working under the global's big company",
                "teja industries": "You're a guy who knows all the leaking problems. So, you're employed",
                "arvind": "You're the guy who sells Audi cars. So you too a employed one",
                "arjun reddy": "You're a big shit and you're jobless",
            }
        }

    def make_decision(self, category, condition):
        normalized_category = category.lower()
        if normalized_category not in self.conditions:
            print(f"Error: '{category}' is not a recognized category.")
            return "No specific decision based on provided conditions."

        decisions = self.conditions[normalized_category]
        normalized_condition = condition.lower()
        decision = decisions.get(normalized_condition, "No specific decision based on provided conditions.")

        if decision == "No specific decision based on provided conditions.":
            print(f"Error: '{condition}' is not a recognized condition in the '{category}' category.")

        return decision

    def perform_kmeans_clustering(self, data, num_clusters):
        # Perform k-means clustering with explicit setting of n_init
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        return labels

# Example usage
decision_maker = DecisionMaker()

# Simulate different conditions
category = input("Enter the category (e.g., 'weather', 'coffee_temperature', 'heart_beats', 'job_status_prediction'): ")
condition = input(f"Enter the condition for {category}: ")

result = decision_maker.make_decision(category, condition)
print("Decision:", result)

# Example of using k-means clustering
data_for_clustering = np.random.rand(100, 2)  # Example dataset, you can replace it with your own data
num_clusters = 3

cluster_labels = decision_maker.perform_kmeans_clustering(data_for_clustering, num_clusters)
print("K-Means Clustering Labels:", cluster_labels)
