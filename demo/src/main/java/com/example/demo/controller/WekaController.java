package com.example.demo.controller;

import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.clusterers.SimpleKMeans;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;

import java.io.InputStream;
import java.io.InputStreamReader;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Files;


public class WekaController {



    @PostMapping("/upload")
    public String analyzeFile(
            @RequestParam("file") MultipartFile file,
            @RequestParam("method") String method,
            @RequestParam(value = "evaluation", required = false) String evaluationMethod) {
        try {
            // Determinar el tipo de archivo (CSV o ARFF) y cargar los datos adecuadamente
            Instances data;
            String fileName = file.getOriginalFilename();
            if (fileName != null && fileName.endsWith(".csv")) {
                data = loadCSV(file.getInputStream());
            } else {
                data = loadARFF(file.getInputStream());
            }

            // Establecer el índice de la clase, si existe
            if (data.classIndex() == -1 && data.numAttributes() > 1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // Llamar al método correspondiente
            String result;
            switch (method) {
                case "clustering":
                    result = performClustering(data);
                    break;
                case "classification":
                    result = performClassification(data, evaluationMethod);
                    break;
                case "kMeans":
                    result = performKMeans(data);
                    break;
                default:
                    result = "Método no reconocido.";
            }

            return result;
        } catch (Exception e) {
            e.printStackTrace();
            return "Error durante el análisis: " + e.getMessage();
        }
    }

    private Instances loadCSV(InputStream inputStream) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(inputStream);
        return loader.getDataSet();
    }

    private Instances loadARFF(InputStream inputStream) throws Exception {
        return new Instances(new InputStreamReader(inputStream));
    }

    private String performClustering(Instances data) {
        try {
            // Eliminar el atributo de clase antes de clustering
            Remove remove = new Remove();
            remove.setAttributeIndices(String.valueOf(data.classIndex() + 1)); // WEKA usa índices 1-based
            remove.setInputFormat(data);
            Instances dataWithoutClass = Filter.useFilter(data, remove);

            // Configurar y aplicar K-Means
            SimpleKMeans kMeans = new SimpleKMeans();
            kMeans.setNumClusters(3); // Número de clusters deseado
            kMeans.buildClusterer(dataWithoutClass);

            // Construir resultados
            StringBuilder result = new StringBuilder("Resultados del Clustering:\n");
            for (int i = 0; i < dataWithoutClass.numInstances(); i++) {
                int cluster = kMeans.clusterInstance(dataWithoutClass.instance(i));
                result.append("Instancia ").append(i).append(" en Cluster ").append(cluster).append("\n");
            }
            return result.toString();
        } catch (Exception e) {
            e.printStackTrace();
            return "Error al realizar el clustering: " + e.getMessage();
        }
    }

    private String performKMeans(Instances data) {
        try {
            // Eliminar el atributo de clase antes de clustering
            Remove remove = new Remove();
            remove.setAttributeIndices(String.valueOf(data.classIndex() + 1)); // WEKA usa índices 1-based
            remove.setInputFormat(data);
            Instances dataWithoutClass = Filter.useFilter(data, remove);

            // Configurar y aplicar K-Means
            SimpleKMeans kMeans = new SimpleKMeans();
            kMeans.setNumClusters(2); // Número de clusters deseado
            kMeans.buildClusterer(dataWithoutClass);

            // Crear el resultado del análisis
            StringBuilder result = new StringBuilder("kMeans\n======\n\n");

            result.append("Number of clusters: ").append(kMeans.getNumClusters()).append("\n");
            result.append("Within cluster sum of squared errors: ").append(kMeans.getSquaredError()).append("\n\n");

            result.append("Final cluster centroids:\n");
            Instances centroids = kMeans.getClusterCentroids();
            for (int i = 0; i < centroids.numInstances(); i++) {
                result.append("Cluster ").append(i).append(": ");
                for (int j = 0; j < centroids.numAttributes(); j++) {
                    result.append(centroids.instance(i).value(j)).append(", ");
                }
                result.append("\n");
            }

            // Calcular manualmente el tamaño de los clústeres
            int[] clusterSizes = new int[kMeans.getNumClusters()];
            for (int i = 0; i < dataWithoutClass.numInstances(); i++) {
                int cluster = kMeans.clusterInstance(dataWithoutClass.instance(i));
                clusterSizes[cluster]++;
            }

            result.append("\nClustered Instances:\n");
            for (int i = 0; i < clusterSizes.length; i++) {
                result.append("Cluster ").append(i).append(": ").append(clusterSizes[i])
                      .append(" (").append((clusterSizes[i] * 100.0 / dataWithoutClass.numInstances())).append("%)\n");
            }

            // Contar las instancias incorrectas
            double incorrectCount = 0;
            for (int i = 0; i < data.numInstances(); i++) {
                // Obtener el clúster al que se asigna la instancia
                int cluster = kMeans.clusterInstance(dataWithoutClass.instance(i));

                // Obtener la clase real de la instancia
                double realClassValue = data.instance(i).classValue();

                // Si el clúster asignado no coincide con la clase real, se considera incorrecto
                if (cluster != realClassValue) {
                    incorrectCount++;
                }
            }

            // Mostrar la cantidad de instancias incorrectamente clasificadas
            result.append("\nIncorrectly clustered instances: ").append(incorrectCount).append(" (")
                  .append((incorrectCount / data.numInstances()) * 100).append("%)\n");

            return result.toString();
        } catch (Exception e) {
            e.printStackTrace();
            return "Error al realizar K-Means: " + e.getMessage();
        }
    }

    private String performClassification(Instances data, String evaluationMethod) {
        try {
            // Crear un clasificador J48 (C4.5)
            J48 j48 = new J48();
            j48.buildClassifier(data);

            // Evaluar el modelo
            Evaluation eval;
            if ("cross-validation".equalsIgnoreCase(evaluationMethod)) {
                eval = new Evaluation(data);
                eval.crossValidateModel(j48, data, 10, new java.util.Random(1));
            } else { // Por defecto: usar conjunto de entrenamiento
                eval = new Evaluation(data);
                eval.evaluateModel(j48, data);
            }

            // Crear un StringBuilder para los resultados
            StringBuilder result = new StringBuilder("Resultados de la Clasificación:\n");

            // Resultado de la evaluación general
            result.append(eval.toSummaryString("\nResultados de la Evaluación\n", false));

            // Detalles de la precisión por clase
            result.append("\n\n=== Detailed Accuracy By Class ===\n");
            result.append(eval.toClassDetailsString());

            // Matriz de confusión
            result.append("\n\n=== Confusion Matrix ===\n");
            result.append(eval.toMatrixString());

            return result.toString();
        } catch (Exception e) {
            e.printStackTrace();
            return "Error al realizar la clasificación: " + e.getMessage();
        }
    }

}
