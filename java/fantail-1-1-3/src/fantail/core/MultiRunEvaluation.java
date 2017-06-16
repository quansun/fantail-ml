/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fantail.core;

import fantail.algorithms.AbstractRanker;
import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Quan Sun quan.sun.nz@gmail.com
 */
public class MultiRunEvaluation {

    final Instances m_Data;
    private Random m_Rand;
    private double[] m_ScoreKendall;
    private double[] m_ScoreSpearmanCC;

    public MultiRunEvaluation(Instances data) {
        m_Data = new Instances(data);
    }

    public double getScoreKendall() {
        return weka.core.Utils.mean(m_ScoreKendall);
    }
    
    public double getScoreSpearmanCC() {
        return weka.core.Utils.mean(m_ScoreSpearmanCC);
    }
    
    public double getScoreSpearmanCCStd() {
        return Math.sqrt(weka.core.Utils.variance(m_ScoreSpearmanCC));
    }
    
    public double getScoreKendallStd() {
        return Math.sqrt(weka.core.Utils.variance(m_ScoreKendall));
    }

    public void multiRunEvaluate(AbstractRanker ranker,
            int numRuns,
            double ratio,
            int randSeed) throws Exception {
        if (numRuns <= 1) {
            throw new Exception("numRuns must be greater than 0.");
        }
        //m_NumRuns = numRuns;
        m_Rand = new Random(randSeed);
        int numTrainInstances = (int) (ratio * m_Data.numInstances());
        //
        m_ScoreKendall = new double[numRuns];
        m_ScoreSpearmanCC = new double[numRuns];
        //
        for (int i = 0; i < numRuns; i++) {
            m_Data.randomize(m_Rand);
            Instances train = new Instances(m_Data, 0);
            Instances test = new Instances(m_Data, 0);
            for (int m = 0; m < m_Data.numInstances(); m++) {
                if (train.numInstances() <= numTrainInstances) {
                    train.add(m_Data.instance(m));
                } else {
                    test.add(m_Data.instance(m));
                }
            }
            ranker.buildRanker(train);
            double localScoreKendall = 0;
            double localScoreSpearmanCC = 0;
            for (int m = 0; m < test.numInstances(); m++) {
                Instance inst = test.instance(m);
                double[] pred = ranker.recommendRanking(inst);
                double[] actual = Tools.getTargetVector(inst);
                localScoreKendall += EvalTools.computeKendallTau(actual, pred);
                localScoreSpearmanCC += EvalTools.computeSpearmanCC(actual, pred);
            }
            localScoreKendall /= test.numInstances();
            localScoreSpearmanCC /= test.numInstances();
            m_ScoreKendall[i] += localScoreKendall;
            m_ScoreSpearmanCC[i] += localScoreSpearmanCC;
        }
    }
}
