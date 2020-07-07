def run_model(classifier, X, y):
    model = classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 15)
    
    # model params
    print(model.fit(X_train, y_train))
    
    # recall scores:
    print(f"Training recall score: {recall_score(y_train, model.predict(X_train))}")
    print(f"Test recall score: {recall_score(y_test, model.predict(X_test))}")
    
    #Cross val scores for recall:
    print(f"Cross val Score train:  {cross_val_score(model, X_train, y_train, cv=5, scoring='recall')}")
    print(f"Cross val Score test:  {cross_val_score(model, X_test, y_test, cv=5, scoring='recall')}")
    
    # Confusion matrix:
    print(f"Train: {confusion_matrix(y_train, model.predict(X_train))}")
    print(f"Test: {confusion_matrix(y_test, model.predict(X_test))}")