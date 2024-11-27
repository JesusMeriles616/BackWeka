package com.example.demo.controller;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToNominal;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;

import weka.filters.unsupervised.attribute.NumericToNominal;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import com.example.demo.controller.WebConfig;


@CrossOrigin(origins = "http://localhost:4200")
@RestController
@RequestMapping("/api/analyze")
public class ApiController {


    @PostMapping("/upload")
    public String analyzeFile(
            @RequestParam("file") MultipartFile file,
            @RequestParam("method") String method,
            @RequestParam(value = "evaluation", required = false) String evaluationMethod) {
        try {
            // Cargar los datos según el tipo de archivo
            Instances data;
            String fileName = file.getOriginalFilename();
            if (fileName != null && fileName.endsWith(".csv")) {
                data = loadCSV(file.getInputStream());
            } else {
                data = loadARFF(file.getInputStream());
            }

            // Validar los datos cargados
            validateData(data);

            // Preprocesar los datos
            data = preprocessData(data);

            // Establecer el índice de la clase, si aplica
            if (data.classIndex() == -1 && data.numAttributes() > 1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // Seleccionar el análisis
            String result;
            switch (method.toLowerCase()) {
                case "clustering":
                    result = performClustering(data);
                    break;
                case "classification":
                    result = performClassification(data, evaluationMethod);
                    break;
                case "kmeans":
                    result = performKMeans(data);
                    break;
                case "neuralnetwork":
                    result = performNeuralNetwork(data, evaluationMethod);
                    break;
                default:
                    result = "Método de análisis no reconocido.";
            }
            return result;
        } catch (Exception e) {
            e.printStackTrace();
            //System.out.println("Dinosaurio");
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
    public ResponseEntity<String> uploadFile(@RequestParam("file") MultipartFile file,
                                             @RequestParam("method") String method,
                                             @RequestParam(value = "evaluation", required = false) String evaluation) {
        try {
            String result = analyzeFile2(file, method, evaluation);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Error processing file: " + e.getMessage());
        }
    }
    public String analyzeFile2(MultipartFile file, String method, String evaluation) throws Exception {
        // Guardar el archivo temporalmente
        Path tempFile = Files.createTempFile(null, null);
        file.transferTo(tempFile.toFile());

        // Cargar instancias desde el archivo
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(tempFile.toString());
        Instances data = source.getDataSet();

        // Validar el conjunto de datos
        validateData(data);

        // Preprocesar los datos
        data = preprocessData(data);

        // Asignar el índice de clase (asegúrate de que esté configurado)
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1); // Por defecto, usa el último atributo como clase
        }

        // Determinar el método de análisis
        String result;
        switch (method) {
            case "classification":
                result = performClassification(data, evaluation);
                break;
            case "clustering":
                result = performClustering(data);
                break;
            case "kMeans":
                result = performKMeans(data);
                break;
            default:
                throw new IllegalArgumentException("Método de análisis no soportado");
        }

        // Eliminar el archivo temporal
        Files.delete(tempFile);

        return result;
    }



    private void validateCSVFormat(BufferedReader reader) throws Exception {
        String line;
        int expectedColumns = -1;
        int lineNumber = 0;

        while ((line = reader.readLine()) != null) {
            lineNumber++;
            String[] values = line.split(",");
            if (expectedColumns == -1) {
                expectedColumns = values.length;
            } else if (values.length != expectedColumns) {
                throw new IllegalArgumentException(
                        "Formato inconsistente en el archivo CSV: línea " + lineNumber + " tiene " +
                                values.length + " valores, se esperaban " + expectedColumns
                );
            }
        }
    }

    private void validateData(Instances data) {
        if (data.numInstances() == 0) {
            throw new IllegalArgumentException("El conjunto de datos está vacío.");
        }
        if (data.numAttributes() < 2) {
            throw new IllegalArgumentException("El conjunto de datos debe tener al menos 2 atributos.");
        }
    }

    private Instances preprocessData(Instances data) throws Exception {
        // Convertir atributos String a Nominal
        data = convertStringToNominal(data);

        // Convertir atributos numéricos a Nominal si es necesario para clasificación
        if (data.classIndex() >= 0 && data.attribute(data.classIndex()).isNumeric()) {
            data = convertNumericToNominal(data);
        }

        return data;
    }

    private Instances convertStringToNominal(Instances data) throws Exception {
        StringToNominal filter = new StringToNominal();
        filter.setInputFormat(data);

        StringBuilder stringAttributes = new StringBuilder();
        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attribute(i).isString()) {
                if (stringAttributes.length() > 0) {
                    stringAttributes.append(",");
                }
                stringAttributes.append(i + 1); // WEKA usa índices 1-based
            }
        }

        if (stringAttributes.length() > 0) {
            filter.setAttributeRange(stringAttributes.toString());
            data = Filter.useFilter(data, filter);
        }

        return data;
    }

    private Instances convertNumericToNominal(Instances data) throws Exception {
        NumericToNominal filter = new NumericToNominal();
        filter.setAttributeIndices(String.valueOf(data.classIndex() + 1)); // WEKA usa índices 1-based
        filter.setInputFormat(data);
        return Filter.useFilter(data, filter);
    }
    private String performNeuralNetwork(Instances data, String evaluationMethod) {
        try {
            System.out.println("Se usa el metodo REdes ");
            // Crear el clasificador de red neuronal
            MultilayerPerceptron mlp = new MultilayerPerceptron();
            // Configurar parámetros de la red
            mlp.setLearningRate(0.1);
            mlp.setMomentum(0.2);
            mlp.setTrainingTime(400); // Número de épocas
            mlp.setHiddenLayers("a"); // Número de nodos en cada capa oculta, 'a' es (atributos + clases) / 2

            // Construir el clasificador con los datos
            mlp.buildClassifier(data);

            // Evaluar el modelo
            Evaluation eval = new Evaluation(data);
            if ("cross-validation".equalsIgnoreCase(evaluationMethod)) {
                eval.crossValidateModel(mlp, data, 10, new java.util.Random(1));
            } else {
                eval.evaluateModel(mlp, data);
            }

            // Retornar resultados de la evaluación
            return eval.toSummaryString("\nResultados de la Red Neuronal\n", false) +
                    eval.toClassDetailsString() +
                    "\n\n=== Confusion Matrix ===\n" +
                    eval.toMatrixString();
        } catch (Exception e) {
            e.printStackTrace();
            return "Error al realizar la red neuronal: " + e.getMessage();
        }
    }
}
