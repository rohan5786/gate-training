public class Neuron {
    private float bias;
    private float[] inputs, weights;
    private final float learningRate = 2f;

    public Neuron(float[] inputs) {
        bias = (float) (2 * Math.random() - 1);
        this.inputs = inputs;
        weights = new float[inputs.length];
        for (int i = 0; i < weights.length; i++)
            weights[i] = (float) (2 * Math.random() - 1); // (-1, 1)
    }
    
    public void setInputs(float[] inputs) {
        this.inputs = inputs;
    }

    // gets activated, weighted sum --> return a = f(z) = sigmoid(sum(w_j * i_j) +
    // b)
    public float getWeightedSum() {
        float sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            sum += weights[i] * inputs[i];
        }
        // activation function
        return sum + bias;
    }
    
    // activation function result
    public float getForward() {
        return Util.sigmoid(getWeightedSum());
    }

    public String toString() {
        String str = "weights: ";
        for(float x : weights) 
            str += x + ", ";
        return str + "\nz: " + getWeightedSum() + ", a: " + getForward();
    }

    // back
    public void learn(float goal) {
        // calulate the dL/dw (how much loss changes based on weight) = dL/da * da/dz *
        // dz/dw
        // dz/dw = input, da/dz = d(sigmoid)/dz = d(sigmoid(z)), dL/da = (input % 2) - a
        // a = sigmoid(z), z = sum(weight * input) + bias

        // Thus:
        // {(goal - a) = dL/da} * {(sigmoid(z) = da/dz} * {(input = dz/dw) = dL/dw}

        final float a = getForward();
        final float z = getWeightedSum(); // sum of weights * inputs + bias
        final float delta = (a - goal) * Util.dSigmoid(z); // 0 is any (even #) % 2

        for (int i = 0; i < inputs.length; i++) {
            weights[i] -= learningRate * delta * inputs[i]; // mult of derivatives to get dL/dw
        }
        bias -= learningRate * delta; // same x * dL/db = dL/dw logic, only once; bias is added once
    }

}