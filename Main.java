import java.lang.reflect.Array;
import java.util.Arrays;

public class Main {

    public static float max(float[] set) {
        Arrays.sort(set);
        return set[set.length - 1];
    }

    public static void main(String[] args) {
        Layer layer = new Layer(), hidden = new Layer();

        float[] dataset = { 1f, 3f, 8f, 4f, 9f, 25f, 16f, 33f, 6767f, 57f, 68f };
        float max = max(dataset);
        float[] results = new float[dataset.length];
        int index = 0;
        float[] inputs = { 0f };

        for (float d : dataset) {
            inputs[0] = (d / max) * 5f;
            layer.setNeurons(inputs);
            layer.setGoal(d % 2);
            hidden.setNeurons(layer.giveForwardInputs());
            hidden.setGoal(layer.goal());
            // layer.updateWeights();

            for (int i = 0; i < 1000000; i++) {
                hidden.updateInputs(layer.giveForwardInputs());
                // hidden.setGoal(layer.goal());
                hidden.updateWeights(); // train hidden
                layer.updateWeights(); // train input
            }

            // layer.setNeurons(hidden.giveForwardInputs());
            results[index] = layer.giveForward();
            index++;
        }
        // System.out.println("\n" + layer.allNeurons());
        index = 0;
        for (float res : results) {
            System.out.println("input: " + dataset[index] + " is "
                    + (res < 0.01f ? "Even" : (res > 0.99f ? "Odd" : "Not trained enough to know")) + " (% 2 = " + res
                    + ")");
            index++;
        }

        // Scanner s = new Scanner(System.in);
        // System.out.println("choose a num");
        // int x = s.nextInt();
        // layer.setNeurons(inputs)
    }
}