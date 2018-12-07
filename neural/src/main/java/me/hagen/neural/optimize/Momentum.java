package me.hagen.neural.optimize;

import me.hagen.neural.optimize.Abstract.GradientFields;
import me.hagen.neural.optimize.Abstract.Gradients;
import me.hagen.neural.optimize.Abstract.Optimizer;
import me.hagen.neural.optimize.Abstract.LearningRate;

public class Momentum implements Optimizer {


	@Override
	public Gradients getDeltaGradient(GradientFields fields, Gradients grad, LearningRate r) {
		MomentumFields m = (MomentumFields) fields;
		m.updateFields(grad);
		Gradients ret = new Gradients(grad.delta.length);
		for(int i =0;i<m.delta.length;i++) {
			ret.delta[i]=m.delta[i].mul(r.rate[i]);
		}
		return ret;
	}

	@Override
	public GradientFields getFields(int paramSize) {
		return new MomentumFields(paramSize);
	}

	@Override
	public LearningRate getLearningRate(int paramSize, double init) {
		return new LearningRate(paramSize, init);
	}
	
	public static class MomentumFields extends Gradients implements GradientFields{
		
		public double rate  = 0.8;
		public MomentumFields(int size) {
			super(size);
		}

		@Override
		public void updateFields(Gradients grad) {
			if(delta[0]==null) {
				for(int i =0;i<grad.delta.length;i++) {
					delta[i]=grad.delta[i].dup();
				}
			}
			else {
				for(int i =0;i<grad.delta.length;i++) {
					delta[i].muli(rate);
					delta[i].addi(grad.delta[i].mul(1-rate));
				}
			}
			
		}
		
	}
}
