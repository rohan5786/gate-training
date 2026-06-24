public class Util {
    public static float sigmoid(float x) {
        return (float) (1 / (1 + Math.exp(-x)));
    }

    public static float dSigmoid(float x) {
        final float sigmoid = sigmoid(x);
        return sigmoid * (1 - sigmoid);
    }
}
