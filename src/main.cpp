#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <experimental/filesystem>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <locale>
#include <vector>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>
#include <iomanip>
#include <ctime>
#include <set>
#include <complex>

namespace fs = std::experimental::filesystem;

const double PI=3.14159265359;

void silent_shell(const char* cmd)
    {
        std::array<char, 128> buffer;
        std::string result;
        std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
        if (!pipe) throw std::runtime_error("popen() failed!");
        while (!feof(pipe.get())) {
            if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
                result += buffer.data();
        }
    }

std::string GetSysResponse(const char* cmd)
    {
        std::array<char, 128> buffer;
        std::string result;
        std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
        if (!pipe) throw std::runtime_error("popen() failed!");
        while (!feof(pipe.get())) {
            if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
                result += buffer.data();
        }
        return result;
    }

bool CheckProgAvailable(const char* program)
    {
            std::string result;
            std::string cmd;
            cmd = "which ";
            cmd += program;
            result=GetSysResponse(cmd.c_str());
            if (result.empty())
            {
                std::cout << "Missing program: " << program << std::endl;
                return false;
            }
            return true;
    }

void usage()
{
    std::cout << "Usage: " << std::endl;
    std::cout << "    tcmdir -i <input file> -o <output file>" << std::endl;
    std::cout << "        -i  |  /path/to/output/from/terachem" << std::endl;
    std::cout << "        -o  |  /path/to/desired/output.csv" << std::endl;
}

void parse_args(int argc, char **argv, std::string &input_file, std::string &output_file)
{
    if (argc < 5)
    {
        usage();
        exit(0);
    }
    for (int i=0; i < argc; i++)
    {   
        if ((std::string)argv[i] == "-i")
        {
            input_file = (std::string)argv[i+1];
            i++;
        }
        else if ((std::string)argv[i] == "-o")
        {
            output_file = (std::string)argv[i+1];
            i++;
        }
        else
        {
            continue;
        }
    }
    if ((input_file == "") || (output_file == ""))
    {
        std::cout << "Unable to parse filenames correctly.  Exiting." << std::endl;
        exit(0);
    }

    std::cout << "Reading " << input_file << " and processing resulting IR spectrum to " << output_file << std::endl;
}

void validate_input_file(std::string input_filename)
{
    if (! fs::exists(input_filename))
    {
        std::cout << "Unable to locate input file: " <<input_filename << std::endl;
        exit(0);
    }

    std::ifstream ifile(input_filename,std::ios::in);
    if (! ifile.is_open())
    {
        std::cout << "Unable to open input file: " <<input_filename << std::endl;
        exit(0);
    }
    ifile.close();
    std::cout << "Input file validated." << std::endl;
}

double get_timestep(std::string input_filename)
{
    // "Velocity Verlet integration time step: 1.00 fs"
    std::ifstream ifile(input_filename,std::ios::in);
    std::string line;
    while (getline(ifile,line))
    {
        if (line.find("Velocity Verlet integration time step:") != std::string::npos)
        {
            std::stringstream dummy;
            line = line.substr(line.find(":")+1,line.find("fs") - line.find(":") - 1);
            double timestep = stod(line);
            ifile.close();
            return timestep;
        }
    }
    std::cout << "Unable to determine timestep.  Exiting." << std::endl;
    exit(0);
}

std::vector <std::vector<double>> process_file_to_dipoles(std::string input_filename)
{
    std::vector <std::vector<double>> dipoles = {};
    std::ifstream ifile(input_filename,std::ios::in);
    std::string line;
    while (getline(ifile,line))
    {
        if (line.find("DIPOLE MOMENT")== std::string::npos)
        {
            continue;
        }
        // "DIPOLE MOMENT: {-1.594580, -7.511072, 5.238981} (|D| = 9.295473) DEBYE"
        std::string chunk = line.substr(line.find("{")+1,line.find("}")-line.find("{")-1);
        while (chunk.find(",")!= std::string::npos)
        {
            int pos = chunk.find(',');
            chunk[pos] = ' ';
        }
        std::stringstream dummy;
        dummy.str(chunk);
        double x, y, z;
        dummy >> x >> y >> z;
        dummy.flush();
        std::vector<double> coords = {x,y,z};
        dipoles.push_back(coords);
    }
    
    if (dipoles.size() == 0)
    {
        std::cout << "No dipole information found.  Exiting. " <<std::endl;
        exit(0);
    }
    return dipoles;
}

std::vector< std::vector<double>> differentiate(std::vector< std::vector<double>> dipoles, double timestep)
{
    std::vector< std::vector<double>> diff={};
    for (int i=0; i < dipoles.size()-1; i ++)
    {
        double dx,dy,dz;
        dx = dipoles[i+1][0] - dipoles[i][0];
        dy = dipoles[i+1][1] - dipoles[i][1];
        dz = dipoles[i+1][2] - dipoles[i][2];
        diff.push_back({dx/timestep,dy/timestep,dz/timestep});
    }
    return diff;
}

std::vector<double> autocorrelation(std::vector<std::vector<double>> data_deriv,int max_dt)
{
    int num_timesteps = data_deriv.size();
    if (max_dt > num_timesteps)
    {
        max_dt = num_timesteps;
    }
    std::vector<double> auto_corr_func(max_dt);
    // auto_corr_func[0] = 1.0;
    for (int i = 0; i < max_dt; i++)
    {
        // Va = data_deriv[: num_timesteps - i, :]  # the first num_timesteps-i rows
        std::vector<std::vector<double>> Va = {};
        for (int j = 0; j < num_timesteps-i; j++)
        {
            Va.push_back(data_deriv[j]);
        }

        //Vb = data_deriv[i:, :]  # ignore the first i rows
        std::vector<std::vector<double>> Vb = {};
        for (int j = i; j < data_deriv.size(); j++)
        {
            Vb.push_back(data_deriv[j]);
        }
        
        // calculate the mean of (Va*Vb)
        double total=0;
        int element_count=0;
        if (Va.size() != Vb.size())
        {
            std::cout << "Mismatch in array sizes in autocorrelation function!" << std::endl;
            std::cout << "Va size: " <<Va.size() << std::endl;
            std::cout << "Vb size: " <<Vb.size() << std::endl;
            exit(0);
        }
        for (int j=0; j < Va.size(); j++)
        {
            total += Va[j][0]*Vb[j][0];
            total += Va[j][1]*Vb[j][1];
            total += Va[j][2]*Vb[j][2];
            element_count += 3;
        }
        auto_corr_func[i] = total/(double)element_count;
    }

    for (int i=0; i < auto_corr_func.size(); i++) //Normalize to first value in array.
    {
        auto_corr_func[i] *= 3;
    }
    return auto_corr_func;
}

std::vector<double> get_dipole_derivatives_acf(std::vector <std::vector<double>> dipoles, double timestep)
{
    std::vector< std::vector<double>> data_deriv;
    data_deriv = differentiate(dipoles,timestep);
    int max_dt = data_deriv.size()/2;
    if (max_dt > 5000)
    {
        max_dt = 5000;
    }
    return autocorrelation(data_deriv, max_dt);
}

std::vector<double> get_times(std::vector<double> acf, double timestep)
{
    std::vector<double> times={};
    for (int i=0; i < acf.size(); i++)
    {
        times.push_back(i*timestep);
    }
    return times;
}

std::vector<double> tapered_cosine_smoothing(std::vector<double> acf, std::vector<double> times)
{
    // def tapered_cosine(x):
    //     return 0.5 * (np.cos(np.pi * x / np.max(x)) + 1)
    std::vector <double> smoothed = {};
    double max_time = times.back();
    for (int i=0; i < acf.size(); i++)
    {
        smoothed.push_back(acf[i] * 0.5 * (std::cos(PI*times[i] / max_time)+1));
    }
    return smoothed;
}

void fast_fourier_transform(std::vector<std::complex<double>> &array, int N)
{
    // recursive Cooley–Tukey FFT
    const double PI=3.14159265359;
    // const size_t N = array.size();
	if (N <= 1) return;

	// divide
	std::vector<std::complex<double>> even = {};
	std::vector<std::complex<double>>  odd = {};
    for (int i=0; i < N; i ++)
    {
        if (i%2) // evaluates to true if odd
        {
            odd.push_back(array[i]);
        }
        else // if above if false, i%2 == 0, so it's even.
        {
            even.push_back(array[i]);
        }
    }

	// conquer
	fast_fourier_transform(even,N/2);
	fast_fourier_transform(odd,N/2);

	// combine
	for (size_t k = 0; k < N/2; k++)
	{
		std::complex<double> t = std::polar(1.0, -2 * PI * k / N) * odd[k];
		array[k    ] = even[k] + t;
		array[k+N/2] = even[k] - t;
	}
}

void power_spectrum(std::vector<double> times, std::vector<double> acf, std::vector<double> &frequencies, std::vector<double> &intensities)
{
    const double speed_of_light = 29979245800.0; // in cm/s
    double dt = (times[1] - times[0]) * 1e-15;
    double max_frequency = 5000.0;
    int n_points = 1 / (dt * speed_of_light);
    if (n_points < acf.size())
    {
        n_points = acf.size();
    }

    // Set frequencies array to values.
    for (int i =0; i < n_points; i++)
    {
        frequencies.push_back((double)i/(dt * speed_of_light * (double) n_points));
    }

    // Generate complex form of autocorrelation function (a+bi) for FFT.
    std::vector<std::complex<double>> complex_intensities(n_points);
    for (int i=0; i < acf.size(); i++)
    {
        complex_intensities[i] = std::complex<double>(acf[i],0.0);
    }
    for (int i=acf.size(); i < n_points; i++)
    {
        complex_intensities[i] = std::complex<double>(0.0,0.0);
    }
    
    // Perform FFT on complex acf to get back intensities.
    fast_fourier_transform(complex_intensities,n_points);

    //    intensities = np.real(intensities[frequencies < max_freq])
    for (int i=0; i < frequencies.size(); i++)
    {
        if (frequencies[i] > max_frequency)
        {
            break;
        }
        
        intensities.push_back(complex_intensities[i].real());
    }
}

void PlotWithPython(std::string outfile_name)
{
    if (!CheckProgAvailable("python"))
    {
        return;
    }
    std::stringstream buffer;
    buffer.str("");
    buffer << "import numpy as np" << std::endl;
    buffer << "import matplotlib.pyplot as plt" << std::endl;
    buffer << "data = np.genfromtxt(\"" << outfile_name << "\", skip_header=1, delimiter=',')" << std::endl;
    buffer << "x_freq = data[:,0]" << std::endl;
    buffer << "y_intens_raw = data[:,1]" << std::endl;
    buffer << "fig = plt.figure(figsize=[8,5],dpi=300,facecolor='white')" << std::endl;
    buffer << "ax = fig.add_subplot(1,1,1)" << std::endl;
    buffer << "## fingerprint region" << std::endl;
    buffer << "ax.axvspan(500,1500,color='grey',alpha=0.25)" << std::endl;
    buffer << "ax.text(1000,max(y_intens_raw)*0.95,'Fingerprint Region',ha='center',va='center',fontsize=8)" << std::endl;
    buffer << "ax.plot(x_freq,y_intens_raw,color='r',linestyle='--',lw=0.5)" << std::endl;
    buffer << "ax.set_xlim(4000,0)" << std::endl;
    buffer << "ax.set_yticks([])" << std::endl;
    buffer << "ax.set_ylim(0,max(y_intens_raw)*1.05)" << std::endl;
    buffer << "ax.set_xlabel(r'Wavenumbers (cm$^{-1}$)')" << std::endl;
    buffer << "fig.tight_layout()" << std::endl;
    buffer << "fig.savefig(\"" << outfile_name.substr(0,outfile_name.size()-4) << ".png\",dpi=300,facecolor='white')" << std::endl;
    buffer << "" << std::endl;
    buffer << "" << std::endl;
    std::ofstream pyfile("plot.py",std::ios::out);
    pyfile << buffer.str();
    pyfile.close();
    silent_shell("python plot.py; rm plot.py");
}

int main(int argc, char **argv)
{
    // Declare variables
    std::string input_filename = "";
    std::string output_filename = "";
    std::vector <std::vector<double>> dipoles = {};
    std::vector<double> autocorr_function;
    std::vector<double> times = {};
    double timestep;
    int n_frames;

    // Parse command line arguments, validate input file's existence.
    parse_args(argc,argv,input_filename,output_filename);
    validate_input_file(input_filename);
    
    // Process input file to xyz of dipole moments.
    dipoles = process_file_to_dipoles(input_filename);

    // get timestep in fs.
    timestep = get_timestep(input_filename);

    // Calculate Autocorrelation Function
    autocorr_function = get_dipole_derivatives_acf(dipoles, timestep);
    std::cout << "Calculated autocorrelation function of dipoles over time." << std::endl;

    // Generate array of times in fs.
    times = get_times(autocorr_function,timestep);

    // Smooth ACF with tapered cosine function.
    autocorr_function = tapered_cosine_smoothing(autocorr_function, times);
    std::cout << "Smoothed autocorrelation with tapered cosine." << std::endl;

    // Calculate intensities @ frequencies.
    std::vector<double> frequencies = {};
    std::vector<double> intensities = {};
    power_spectrum(times, autocorr_function, frequencies, intensities);
    std::cout << "Calculated IR spectrum and frequencies." << std::endl;

    // Save wavenumbers and intensities to CSV file.
    std::ofstream ofile(output_filename,std::ios::out);
    ofile << "Wavenumber(cm-1), Intensity" << std::endl;
    for (int i=0; i < intensities.size(); i++)
    {
        ofile << frequencies[i] << ", " << intensities[i] << std::endl;
    }
    ofile.close();

    std::cout << "IR Spectral data has been written to " << output_filename << std::endl;

    PlotWithPython(output_filename);
    return 0;
}