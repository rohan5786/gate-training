
public class Layer {
    private Neuron[] neurons;
    private float goal;

    public Layer() {
        neurons = new Neuron[5];
    }

    public Layer(float goal) {
        neurons = new Neuron[5];
        this.goal = goal;
    }

    public float goal() {
        return goal;
    }

    public void setNeurons(float[] inputs) {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i] = new Neuron(inputs);
        }
    }

    public void setGoal(float goal) {
        this.goal = goal;
    }

    /*
     * all forward propagation:
     * getting the weighted sum from the last layer, then pass it through the
     * activation function and
     * give the activationOutputs
     */
    public float[] giveForwardInputs() {
        float[] outputs = new float[neurons.length];
        for (int i = 0; i < neurons.length; i++)
            outputs[i] = neurons[i].getForward();
        return outputs;
    }

    public float giveForward() {
        float sum = 0;
        for (int i = 0; i < neurons.length; i++)
            sum += neurons[i].getForward();
        return sum / neurons.length; // avg
    }

    // backward propogation, updating weights + bias
    public void updateWeights() {
        // final float goalPer = goal / neurons.length;
        for (Neuron n : neurons)
            n.learn(goal);
    }

    public void updateInputs(float[] inputs) {
        for(Neuron n : neurons)
            n.setInputs(inputs);
    }

    public String toString() {
        return "" + giveForward();
    }

    public String allNeurons() {
        String ting = "";
        for(int i = 1; i <= neurons.length; i++)
            ting += i + "\n" + neurons[i-1].toString() + '\n';
        return ting + "\n\n";
    }
}
