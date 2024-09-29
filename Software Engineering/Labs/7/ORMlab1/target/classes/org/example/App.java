package org.example;

import java.util.List;
import java.util.Random;
import javax.persistence.criteria.CriteriaBuilder;
import javax.persistence.criteria.CriteriaQuery;
import java.util.Scanner;

import org.hibernate.HibernateException;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.boot.registry.StandardServiceRegistryBuilder;
import org.hibernate.cfg.Configuration;
import org.hibernate.service.ServiceRegistry;

public class App
{
    // some constants for the entities
    private static final String[] firstnames = new String[]{"Kobi", "Almog", "Nisim", "Alex", "Yohai", "Moti", "Yarden", "Eitan", "Bob", "Ofir", "Issac", "Tom"};
    private static final String[] lastnames = new String[]{"Cohen", "Jorno", "Rosenblum", "Hasson", "Peretz", "Shushan", "Avraham", "Levi", "Rosenbaum", "Barazani", "Keler"};
    private static final String characters = "abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    private static final String[] addresses = new String[]{"Haifa", "Tel Aviv", "Carmiel", "Acre", "Eilat", "Ashdod", "Rehovot", "Jerusalem"};

    private static Session session = null;

    // generate password for people
    private static String generatePassword(int length)
    {
        StringBuilder randomString = new StringBuilder(length);
        Random RANDOM = new Random();

        for (int i = 0; i < length; i++)
        {
            // Get a random character from the alphabet
            int randomIndex = RANDOM.nextInt(characters.length());
            char randomChar = characters.charAt(randomIndex);
            // Append the random character to the string
            randomString.append(randomChar);
        }
        return randomString.toString();
    }

    private static String generateEmail(String firstname, String lastname)
    {
        Random RANDOM = new Random();
        return firstname + "." + lastname + RANDOM.nextInt(10) + "@gmail.com";
    }

    private static void generatePeople() throws Exception
    {
        Random random = new Random();

        for (int i = 0; i < 8; i++)
        {
            //init current person
            String firstname = firstnames[random.nextInt(firstnames.length)];
            String lastname = lastnames[random.nextInt(lastnames.length)];
            Person currentPerson = new Person(firstname, lastname,
                    generatePassword(4 + random.nextInt(6)),
                    generateEmail(firstname, lastname));

            session.save(currentPerson);
            session.flush();
        }
    }

    private static String generatePhoneNumber()
    {
        Random RANDOM = new Random();
        return "05" + RANDOM.nextInt(10) + "-" + RANDOM.nextInt(10000000);
    }

    private static String generateAddress()
    {
        Random RANDOM = new Random();
        return addresses[RANDOM.nextInt(addresses.length)] + " " + RANDOM.nextInt(200);
    }

    private static void generateGarages() throws Exception
    {
        Random random = new Random();

        for (int i = 0; i < 2; i++)
        {
            Garage currentGarage = new Garage(generatePhoneNumber(), generateAddress());
            session.save(currentGarage);
            session.flush();
        }
    }



    private static SessionFactory getSessionFactory(String password) throws
            HibernateException
    {
        Configuration configuration = new Configuration();
        configuration.setProperty("hibernate.connection.password", password);

        // Add ALL of your entities here. You can also try adding a whole package.
        configuration.addAnnotatedClass(Car.class);
        configuration.addAnnotatedClass(Person.class);
        configuration.addAnnotatedClass(Garage.class);
        configuration.addAnnotatedClass(Image.class);

        ServiceRegistry serviceRegistry = new
                StandardServiceRegistryBuilder()
                .applySettings(configuration.getProperties())
                .build();

        return configuration.buildSessionFactory(serviceRegistry);
    }

    private static void generateCars() throws Exception
    {
        Random random = new Random();
        List<Person> people = getAllPeople();

        for (int i = 0; i < people.size(); i++)
        {
            Car car = new Car("MOO-" + random.nextInt(), 100000, 2000 + random.nextInt(19),
                    people.get(i),
                    people.get(i).getFirstname()
            );

            // assigns garage list to the car
            assignGarages(car);

            people.get(i).addCar(car);

            session.save(people.get(i));
            session.flush();
            session.save(car);
            session.flush();
        }
    }

    // for every garage - assign at least one owner
    private static void generateGarageOwners() throws Exception
    {
        Random random = new Random();
        List<Garage> garages = getAllGarages();
        List <Person> people = getAllPeople();

        int[] peopleIndices = new int[people.size()];
        boolean allZero = false;

        for(Garage garage: garages)
        {
            // generate 0 or 1 to each cell - 1 the ith person owns the current garage
            for(int i = 0; i < peopleIndices.length; i++)
                peopleIndices[i] = random.nextInt(2);

            int zeroCount = 0;
            for(int i = 0; i < peopleIndices.length; i++)
            {
                if(peopleIndices[i] == 0)
                    zeroCount++;
            }

            // if for some reason no index gets a 1, pick a random index to be a 1
            if(zeroCount == peopleIndices.length)
                peopleIndices[random.nextInt(peopleIndices.length)]++;

            // add selected garages to the car (where cells are 1)
            for(int i = 0; i < peopleIndices.length; i++)
            {
                if(peopleIndices[i] == 1)
                {
                    garage.addOwner(people.get(i));
                    session.save(garage);
                    session.flush();
                    session.save(people.get(i));
                    session.flush();
                }
            }
        }
    }

    // randomly assigns garages to a car
    private static void assignGarages(Car car) throws Exception
    {
        Random random = new Random();
        List<Garage> garages = getAllGarages();

        int[] garagesIndices = new int[garages.size()];
        boolean allZero = false;

        // generate 0 or 1 to each cell - 1 means we include the garage
        for(int i = 0; i < garagesIndices.length; i++)
            garagesIndices[i] = random.nextInt(2);

        int zeroCount = 0;
        for(int i = 0; i < garagesIndices.length; i++)
        {
            if(garagesIndices[i] == 0)
                zeroCount++;
        }

        // if for some reason no index gets a 1, pick a random index to be a 1
        if(zeroCount == garagesIndices.length)
            garagesIndices[random.nextInt(garagesIndices.length)]++;

        // add selected garages to the car (where cells are 1)
        for(int i = 0; i < garagesIndices.length; i++)
        {
            if(garagesIndices[i] == 1)
            {
                car.addGarage(garages.get(i));
                garages.get(i).addCar(car);
                session.save(car);
                session.flush();
                session.save(garages.get(i));
                session.flush();
            }
        }
    }

    private static List<Car> getAllCars() throws Exception
    {
        CriteriaBuilder builder = session.getCriteriaBuilder();
        CriteriaQuery<Car> query = builder.createQuery(Car.class);
        query.from(Car.class);
        List<Car> data = session.createQuery(query).getResultList();
        return data;
    }

    private static List<Person> getAllPeople() throws Exception
    {
        CriteriaBuilder builder = session.getCriteriaBuilder();
        CriteriaQuery<Person> query = builder.createQuery(Person.class);
        query.from(Person.class);
        List<Person> data = session.createQuery(query).getResultList();
        return data;
    }

    private static List<Garage> getAllGarages() throws Exception
    {
        CriteriaBuilder builder = session.getCriteriaBuilder();
        CriteriaQuery<Garage> query = builder.createQuery(Garage.class);
        query.from(Garage.class);
        List<Garage> data = session.createQuery(query).getResultList();
        return data;
    }

    private static void printAllCars() throws Exception
    {
        List<Car> cars = getAllCars();
        for (Car car : cars)
        {
            System.out.print("Id: ");
            System.out.print(car.getId());
            System.out.print(", License plate: ");
            System.out.print(car.getLicensePlate());
            System.out.print(", Price: ");
            System.out.print(car.getPrice());
            System.out.print(", Year: ");
            System.out.print(car.getYear());
            System.out.print('\n');
        }
    }

    private static void printGarages() throws Exception
    {
        List<Garage> garages = getAllGarages();
        int garageCount = 0;
        for (Garage garage : garages)
        {
            garageCount++;
            System.out.println("======== Garage "+garageCount+ " ========");
            int count = 1;
            System.out.println("Phone number: "+ garage.getPhoneNumber());
            System.out.println("Address: "+ garage.getAddress());
            for(Person owner: garage.getOwners())
            {
                System.out.println("Owner " +count+": " + owner.getFirstname() +" "+ owner.getLastname());
                count++;
            }

            count=1;
            for(Car car: garage.getCars())
            {
                System.out.println("Car "+count+" license plate: "+ car.getLicensePlate());
                count++;
            }
            System.out.print('\n');
        }
    }

    private static void printCars() throws Exception
    {
        List<Car> cars = getAllCars();
        int carCount = 0;
        for (Car car : cars)
        {
            carCount++;
            System.out.println("========== Car "+carCount+ " ==========");
            // print car properties
            car.printCarProperties();
            //print owner properties
            car.getOwner().printPersonProperties();

            int garageCount = 1;
            for(Garage garage: car.getAvailableGarages())
            {
                System.out.println("Eligible garage " +garageCount+ " address: " +garage.getAddress());
                garageCount++;
            }
            System.out.print('\n');
        }
    }

    public static void main(String[] args)
    {
        try
        {
            Scanner scanner = new Scanner(System.in);
            System.out.println("Enter password for the connection to the database:");
            String password = scanner.nextLine();
            SessionFactory sessionFactory = getSessionFactory(password);
            session = sessionFactory.openSession();
            session.beginTransaction();

            generatePeople();
            generateGarages();
            generateCars();
            generateGarageOwners();

            printAllCars();
            System.out.println("___________________________________");

            printGarages();
            System.out.println("___________________________________");
            printCars();


            session.getTransaction().commit(); // Save everything.

        } catch (Exception exception)
        {
            if (session != null)
            {
                session.getTransaction().rollback();
            }
            System.err.println("An error occured, changes have been rolled back.");
            exception.printStackTrace();
        } finally
        {
            assert session != null;
            session.close();
        }
    }
}