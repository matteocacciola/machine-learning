interface Point {
  coordinates: number[];
  category?: number;
}

interface Cluster {
  centroid: number[];
  points: Point[];
}

export class FuzzyCMeans {
  private clusters: Cluster[];

  constructor(
    points: Point[],
    private readonly k: number,
    private readonly fuzziness: number,
    private readonly maxIterations: number,
    private readonly epsilon: number
  ) {
    this.clusters = this.initialize(points);
  }

  private initialize(points: Point[]): Cluster[] {
    const max = points.length - 1;
    const clusters: Cluster[] = [];

    for (let i = 0; i < this.k; i++) {
      const randomIndex = Math.floor(Math.random() * max);
      const centroid = points[randomIndex].coordinates;
      clusters.push({ centroid, points: [] });
    }

    return clusters;
  }

  private distance(point: Point, centroid: number[]): number {
    const { coordinates } = point;
    return Math.sqrt(coordinates.reduce((acc, e, i) => acc + (e - centroid[i])**2, 0));
  }

  private membership(point: Point, centroid: number[], clusters?: Cluster[]): number {
    const distances = (clusters ?? this.clusters).map((cluster) => this.distance(point, cluster.centroid));
    return 1 / distances.reduce((sum: number, distance: number) =>
      sum + Math.pow(distance / this.distance(point, centroid), 2 / (this.fuzziness - 1)), 0);
  }

  private assign(points: Point[]): void {
    points.map((point: Point) => {
      const membershipValues = this.clusters.map((cluster) => this.membership(point, cluster.centroid));
      const sum = membershipValues.reduce((acc, value) => acc + value, 0);

      membershipValues.map((membershipValue, index) => {
        point.category = membershipValue / sum;
        this.clusters[index].points.push(point);
      });
    });
  }

  private newCentroids(): number[][] {
    const numDimensions = this.clusters[0].centroid.length;
    return this.clusters.map((cluster) => {
      const numerator = new Array(numDimensions).fill(0);
      let denominator = cluster.points.reduce((acc, point) => {
        const membershipValue = Math.pow(point.category!, this.fuzziness);

        for (let j = 0; j < numDimensions; j++) {
          numerator[j] += membershipValue * point.coordinates[j];
        }

        return acc + membershipValue;
      }, 0);

      return numerator.map((value) => value / denominator);
    });
  }

  private hasConverged(newCentroids: number[][]): boolean {
    const numDimensions = this.clusters[0].centroid.length;
    for (let i = 0; i < this.clusters.length; i++) {
      const centroid = this.clusters[i].centroid;
      const newCentroid = newCentroids[i];
      for (let j = 0; j < numDimensions; j++) {
        if (Math.abs(newCentroid[j] - centroid[j]) > this.epsilon) {
          return false;
        }
      }
    }
    return true;
  }


  public train(points: Point[]): Cluster[] {
    let iterations = 0;
    while (iterations < this.maxIterations) {
      this.assign(points);
      const newCentroids = this.newCentroids();
      if (this.hasConverged(newCentroids)) {
        break;
      }
      this.clusters = this.clusters.map(
        (cluster: Cluster, index: number) => ({ centroid: newCentroids[index], points: [] })
      );
      iterations++;
    }

    return this.clusters;
  }

  public evaluate(points: Point[], clusters: Cluster[]): Point[] {
    return points.map((point: Point) => {
      const membershipValues = clusters.map(cluster => this.membership(point, cluster.centroid, clusters));
      point.category = membershipValues.indexOf(Math.max(...membershipValues));

      return point;
    });
  }
}

/**
 * EXAMPLE USAGE
 *
 * The constructor takes in the points, k, m, maxIterations, and epsilon parameters, where points is an array of
 * Point objects representing the n-dimensional points, k is the number of clusters to form, m is the fuzziness
 * parameter, maxIterations is the maximum number of iterations to perform, and epsilon is the convergence threshold.
 *
 * Once the clusters are initialized, the train function considers the clusters and assign the points to clusters,
 * returning an array of Cluster objects representing the finalized clusters.
 *
 * The evaluate function takes in the points and clusters parameters, where points is an array of Point objects
 * representing the n-dimensional points to be classified, and clusters is an array of Cluster objects representing the
 * clusters obtained from the training phase. It assigns a category to each point in the points array based on its
 * membership values to each cluster in the clusters array.

const points: Point[] = [
  { coordinates: [1, 2, 3], category: 0 },
  { coordinates: [4, 5, 6], category: 0 },
  { coordinates: [7, 8, 9], category: 0 },
  { coordinates: [10, 11, 12], category: 1 },
  { coordinates: [13, 14, 15], category: 1 },
];
const k = 2;
const m = 2;
const maxIterations = 100;
const epsilon = 0.001;

const fuzzyCMeans = new FuzzyCMeans(points, k, m, maxIterations, epsilon);

const clusters = fuzzyCMeans.train(points);

const testPoints: Point[] = [
  { coordinates: [3, 4, 5] },
  { coordinates: [6, 7, 8] },
];

fuzzyCMeans.evaluate(testPoints, clusters);

console.log({ clusters, testPoints });

 */