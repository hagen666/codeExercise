package me.hagen.neural.optimize;

import me.hagen.neural.optimize.Abstract.GradientFields;
import me.hagen.neural.optimize.Abstract.Gradients;
import me.hagen.neural.optimize.Abstract.Optimizer;
import me.hagen.neural.optimize.Abstract.LearningRate;

public class Adagrad implements Optimizer {

    public double rate = 0.8;

    @Override
    public Gradients getDeltaGradient(GradientFields fields, Gradients grad, LearningRate r) {
        AdagradFields m = (AdagradFields) fields;
        m.updateFields(grad);
        Gradients ret = new Gradients(grad.delta.length);
        for (int i = 0; i < grad.delta.length; i++) {
            ret.delta[i] = grad.delta[i].mul(r.rate[i] / Math.sqrt(m.delta[i]));
        }
        return ret;
    }

    @Override
    public GradientFields getFields(int paramSize) {
        return new AdagradFields(paramSize);
    }

    @Override
    public LearningRate getLearningRate(int paramSize, double init) {
        return new LearningRate(paramSize, init);
    }

    public static class AdagradFields implements GradientFields {
        double[] delta = null;

        public AdagradFields(int size) {
            delta = new double[size];
            for (int i = 0; i < delta.length; i++) {
                delta[i] = 0.000001;
            }
        }

        @Override
        public void updateFields(Gradients grad) {
            for (int i = 0; i < grad.delta.length; i++) {
                delta[i] += grad.delta[i].norm1Number().doubleValue();
            }

        }

    }
}
