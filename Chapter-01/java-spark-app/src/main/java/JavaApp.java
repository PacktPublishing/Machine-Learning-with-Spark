import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.DoubleFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * A simple Spark app in Java
 */
public class JavaApp {

    public static void main(String[] args) {
        JavaSparkContext sc = new JavaSparkContext("local[2]", "First Spark App");
        // we take the raw data in CSV format and convert it into a set of records of the form (user, product, price)
        JavaRDD<String[]> data = sc.textFile("data/UserPurchaseHistory.csv")
                .map(new Function<String, String[]>() {
                    @Override
                    public String[] call(String s) throws Exception {
                        return s.split(",");
                    }
                });

        // let's count the number of purchases
        long numPurchases = data.count();

        // let's count how many unique users made purchases
        long uniqueUsers = data.map(new Function<String[], String>() {
            @Override
            public String call(String[] strings) throws Exception {
                return strings[0];
            }
        }).distinct().count();

        // let's sum up our total revenue
        double totalRevenue = data.map(new DoubleFunction<String[]>() {
            @Override
            public Double call(String[] strings) throws Exception {
                return Double.parseDouble(strings[2]);
            }
        }).sum();

        // let's find our most popular product
        // first we map the data to records of (product, 1) using a PairFunction
        // and the Tuple2 class.
        // then we call a reduceByKey operation with a Function2, which is essentially the sum function
        List<Tuple2<String, Integer>> pairs = data.map(new PairFunction<String[], String, Integer>() {
            @Override
            public Tuple2<String, Integer> call(String[] strings) throws Exception {
                return new Tuple2(strings[1], 1);
            }
        }).reduceByKey(new Function2<Integer, Integer, Integer>() {
            @Override
            public Integer call(Integer integer, Integer integer2) throws Exception {
                return integer + integer2;
            }
        }).collect();

        // finally we sort the result. Note we need to create a Comparator function,
        // that reverses the sort order.
        Collections.sort(pairs, new Comparator<Tuple2<String, Integer>>() {
            @Override
            public int compare(Tuple2<String, Integer> o1, Tuple2<String, Integer> o2) {
                return -(o1._2() - o2._2());
            }
        });
        String mostPopular = pairs.get(0)._1();
        int purchases = pairs.get(0)._2();

        // print everything out
        System.out.println("Total purchases: " + numPurchases);
        System.out.println("Unique users: " + uniqueUsers);
        System.out.println("Total revenue: " + totalRevenue);
        System.out.println(String.format("Most popular product: %s with %d purchases",
                mostPopular, purchases));
    
    sc.stop();
    
    }

}
