const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
console.log(xs)

async function loadModel() {
    model = await tf.loadLayersModel("./model.json")
}

loadModel()
