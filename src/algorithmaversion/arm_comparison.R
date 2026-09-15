library(arules)
library(arulesViz)
library(conflicted)

# Tell R to always use the arules version of sorting to prevent errors
conflicts_prefer(arules::sort) 

# Ensure your working directory is set to the folder containing this file
# (Session -> Set Working Directory -> Choose Directory)

file_path <- "cleaned_data_with_labels.csv"

articles <- read.transactions(
  file = file_path,
  format = "basket",
  sep = ",",
  rm.duplicates = FALSE 
)

# These are the "noise" words that clutter the graph without adding insight
words_to_remove <- c("also", "call", "called", "could", "else", "got", 
                     "gotcha", "hey", "lot", "ohh", "side", "totally", "kind", "would", "yeah",
                     "really", "yeah", "like", "well", "definitely")

# Keep only the words that are NOT in the removal list
clean_articles <- articles[, !(itemLabels(articles) %in% words_to_remove)]

targeted_rules <- apriori(
  clean_articles,
  parameter = list(
    support = 0.15,  # Must appear in at least 0.5% of conversations
    confidence = 0.6, # Rule must be at least 20% reliable
    minlen = 2, 
    maxlen = 3
  ),
  # Force the machine to only look for paths leading to our two specific labels
  appearance = list(default = "lhs", rhs = c("ALGORITHM", "SHARING"))
)

sharing_rules <- subset(targeted_rules, subset = rhs %in% "SHARING")
algorithm_rules <- subset(targeted_rules, subset = rhs %in% "ALGORITHM")

# Sort both piles by "Lift" to push the most surprising, meaningful connections to the top
sharing_by_lift <- sort(sharing_rules, by = "lift", decreasing = TRUE)
algorithm_by_lift <- sort(algorithm_rules, by = "lift", decreasing = TRUE)

arules::inspect(head(sharing_by_lift, 15))

arules::inspect(head(algorithm_by_lift, 15))


# Graph 1: The SHARING Web
plot(head(sharing_by_lift, 20), method = "graph", main = "Network: SHARING Recommendations")

# Graph 2: The ALGORITHM Web
plot(head(algorithm_by_lift, 20), method = "graph", main = "Network: ALGORITHM Recommendations")
