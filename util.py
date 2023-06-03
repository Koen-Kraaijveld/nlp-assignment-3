
def load_categories(file_path):
    categories = []
    with open(file_path) as file:
        for category in file:
            categories.append(category.strip())
    return categories
