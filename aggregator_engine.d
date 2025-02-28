import std.stdio;

int aggregateData(int[] data) {
    int sum = 0;
    foreach (num; data) {
        writeln("Adding: ", num); // Debug statement
        sum += num;
    }
    writeln("Final Sum: ", sum); // Debug statement
    return sum;
}
