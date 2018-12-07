package me.hagen.neural.optimize;

import me.hagen.neural.optimize.Abstract.GradientFields;
import me.hagen.neural.optimize.Abstract.Gradients;
import me.hagen.neural.optimize.Abstract.Optimizer;
import me.hagen.neural.optimize.Abstract.LearningRate;

public class SGD implements Optimizer {

	@Override
	public Gradients getDeltaGradient(GradientFields fields, Gradients grad, LearningRate r) {
		fields.updateFields(grad);
		Gradients g = new Gradients(grad.delta.length);
		for(int i = 0;i<grad.delta.length;i++) {
			g.delta[i]=grad.delta[i].mul(r.rate[i]);
		}
		return g;
	}

	@Override
	public GradientFields getFields(int paramSize) {
		return new SGDGradientFields();
	}

	@Override
	public LearningRate getLearningRate(int paramSize,double init) {
		return new LearningRate(paramSize,init);
	}

	public static class SGDGradientFields implements GradientFields{

		@Override
		public void updateFields(Gradients grad) {

		}
		
	}
}
