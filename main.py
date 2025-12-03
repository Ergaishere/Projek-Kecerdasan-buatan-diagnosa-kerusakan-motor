import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt




data = pd.read_csv("data_motor.csv")
print("=== DATA AWAL ===")
print(data)
print()




X = data.drop("kerusakan", axis=1)
y = data["kerusakan"]




X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    min_samples_leaf=1,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("=== AKURASI MODEL ===")
print("Akurasi:", acc)
print()


print("=== SISTEM DIAGNOSA KERUSAKAN MOTOR ===")
brebet = int(input("Motor brebet? (1 ya, 0 tidak): "))
mati_tiba = int(input("Motor mati tiba-tiba? (1 ya, 0 tidak): "))
lampu_redup = int(input("Lampu redup? (1 ya, 0 tidak): "))
boros = int(input("Boros BBM? (1 ya, 0 tidak): "))
susah_starter = int(input("Susah starter? (1 ya, 0 tidak): "))
knalpot_nembak = int(input("Knalpot nembak? (1 ya, 0 tidak): "))

gejala_user = [[
    brebet,
    mati_tiba,
    lampu_redup,
    boros,
    susah_starter,
    knalpot_nembak
]]


hasil = model.predict(gejala_user)
print("\n>> HASIL DIAGNOSA :", hasil[0])


plt.figure(figsize=(18, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=sorted(y.unique()),
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Pohon Keputusan Diagnosa Kerusakan Motor")
plt.show()