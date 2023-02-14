def train(model, all_x, all_y, epochs):

    for epoch in range(epochs):
        error_count = 0

        for x, y in zip(all_x, all_y):
            error = model.update(x, y)
            error_count += abs(error)
        if error_count == 0:
            break

        print(f"Epoch {epoch+1} errors {error_count}")