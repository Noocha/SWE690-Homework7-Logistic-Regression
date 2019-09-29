import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.IOException;
public class LogisticRegressionHW {
    public static void main(String[] args) {

        System.out.println("##################### Have credit history ###################");
//        with history
        String training_dataSet_Filename = "src/creditRisk_Clean_HaveCreditHistory_training.arff";
        String testing_dataSet_Filename = "src/creditRisk_Clean_HaveCreditHistory_testing.arff";
        String predict_dataSet_Filename = "src/creditRisk_Clean_HaveCreditHistory_predict.arff";
        process(training_dataSet_Filename, testing_dataSet_Filename, predict_dataSet_Filename,10);


        System.out.println("###################################################################################");


        System.out.println("###################### No credit history #####################");
//        without history
        training_dataSet_Filename = "src/creditRisk_Clean_NoCreditHistory_training.arff";
        testing_dataSet_Filename = "src/creditRisk_Clean_NoCreditHistory_testing.arff";
        predict_dataSet_Filename = "src/creditRisk_Clean_NoCreditHistory_predict.arff";
        process(training_dataSet_Filename, testing_dataSet_Filename, predict_dataSet_Filename,9);

    }

    public static void process (String trainingFile, String testingFile, String predictFile, int classIndex) {
        Instances trainingDataSet = getDataSet(trainingFile, classIndex);
        Instances testingDataSet = getDataSet(testingFile, classIndex);

        Classifier classifier = new Logistic();
        try {
            classifier.buildClassifier(trainingDataSet);
            Evaluation eval = new Evaluation(trainingDataSet);
            eval.evaluateModel(classifier, testingDataSet);
            System.out.println("Logistics Regression Evaluation with Dataset");
            System.out.println(eval.toSummaryString());
            System.out.println("Logistics Regression Equation");
            System.out.println(classifier);
            System.out.println("Prediction");
            Instances predictDataSet = getDataSet(predictFile, classIndex);
            for (int i = 0; i < predictDataSet.numInstances(); i++) {

                double isMarried = predictDataSet.instance(i).value(0);
                String married = isMarried == 0 ? "Yes" : "No";
                System.out.println("Married status: " + married);

                double dependents = predictDataSet.instance(i).value(1);
                String dependent = dependents == 3 ? "3+" : String.valueOf(dependents);
                System.out.println("Dependents is: " + dependent);

                String graduate = predictDataSet.instance(i).stringValue(2);
                System.out.println("Graduate is: " + graduate);

                double value = classifier.classifyInstance(predictDataSet.instance(i));
                System.out.println(value == 0 ? "You can Loan." : "You can not Loan");


                System.out.println();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static Instances getDataSet(String fileName, int classIndex) {
        ArffLoader loader = new ArffLoader();
        try {
            loader.setFile(new File(fileName));
            Instances dataSet = loader.getDataSet();
            dataSet.setClassIndex(classIndex);
            return dataSet;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

}
