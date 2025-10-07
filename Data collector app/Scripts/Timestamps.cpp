#include <Windows.h>    // Access to Windows API functions
#include <stdio.h>      // Standard I/O operations
#include <fstream>
#include <chrono>
#include <ctime>
#include <mutex>
#include <iostream>
#include <filesystem>
#include <thread>
#include <string>
#include <codecvt>
#include <locale>


std::ofstream outputFile;
std::mutex fileMutex;

// Global variable to store the current foreground window
static HWND g_currentForegroundWindow = NULL;

// Check if a directory exists.
bool directoryExists(const std::string &dirPath)
{
    DWORD attr = GetFileAttributesA(dirPath.c_str());
    if (attr == INVALID_FILE_ATTRIBUTES) {
        return false; // doesn’t exist
    }
    return (attr & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

/**
 * Create a directory if it doesn’t exist.
 * Returns true if the directory now exists (created or already existed).
 */
bool createDirectoryIfNotExists(const std::string &dirPath)
{
    if (directoryExists(dirPath)) {
        return true; // It already exists
    }
    // Try to create directory
    if (CreateDirectoryA(dirPath.c_str(), NULL) == 0) {
        DWORD err = GetLastError();
        if (err != ERROR_ALREADY_EXISTS) {
            return false;
        }
    }
    return true;
}

// Check if a file exists and is not a directory.
bool fileExists(const std::string &filePath)
{
    DWORD attr = GetFileAttributesA(filePath.c_str());
    if (attr == INVALID_FILE_ATTRIBUTES) {
        return false; // No such file
    }
    // If it’s a directory, then it’s not a file
    return (attr & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

// Function to get the executable's directory
std::string getExecutableDirectory()
{
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH); // e.g., "C:\path\to\App.exe"
    
    // Strip off the filename part, leaving only the directory part.
    std::string fullPath(buffer);
    size_t pos = fullPath.find_last_of("\\/");
    return (pos == std::string::npos) ? fullPath : fullPath.substr(0, pos);
}

// Function to print timestamp in [YYYY-MM-DD HH:MM:SS:ms]
void PrintTimestamp()
{
    SYSTEMTIME st;
    GetLocalTime(&st);
    printf("[%04d-%02d-%02d %02d:%02d:%02d:%03d] ",
        st.wYear, st.wMonth, st.wDay,
        st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
    std::lock_guard<std::mutex> lock(fileMutex);
    outputFile << "[" << st.wYear << "-" << st.wMonth << "-" << st.wDay << " "
               << st.wHour << ":" << st.wMinute << ":" << st.wSecond << ":" << st.wMilliseconds << "] ";
}

// Helper function to log window title
void LogWindowTitle(HWND hwnd, const char* prefix)
{
    wchar_t windowTitle[256];
    GetWindowTextW(hwnd, windowTitle, sizeof(windowTitle) / sizeof(windowTitle[0]));

    char titleBuffer[1024];
    int result = WideCharToMultiByte(CP_UTF8, 0, windowTitle, -1, titleBuffer, sizeof(titleBuffer), NULL, NULL);

    if (result > 0) {
        // Print the prefix and converted title using printf
        printf("%s: %s\n", prefix, titleBuffer);

        // Log the title to the file
        std::lock_guard<std::mutex> lock(fileMutex);
        outputFile << prefix << ": " << titleBuffer << std::endl;
    }
}


// Function to check if a UTF-8 string contains "Google Chrome" or "Edge"
bool IsBrowserWindow(HWND hwnd)
{
    wchar_t windowTitle[256];
    GetWindowTextW(hwnd, windowTitle, sizeof(windowTitle) / sizeof(windowTitle[0]));

    char titleBuffer[1024];
    int result = WideCharToMultiByte(CP_UTF8, 0, windowTitle, -1, titleBuffer, sizeof(titleBuffer), NULL, NULL);
    if (result > 0) {
        std::string title(titleBuffer);
        if (title.find("Google Chrome") != std::string::npos || title.find("Edge") != std::string::npos) {
            return true;
        }
    }
    return false;
}


// Callback that will be called when changing main window focus or when an object's name changes
void CALLBACK WinEventProc(HWINEVENTHOOK hHook, DWORD event, HWND hwnd, LONG idObject, LONG idChild, DWORD dwEventThread, DWORD dwmsEventTime)
{
    if (event == EVENT_SYSTEM_FOREGROUND) {
        // Foreground window changed
        g_currentForegroundWindow = hwnd; // Store the new foreground window handle
        PrintTimestamp();
        LogWindowTitle(hwnd, "Window Focus Changed");
    }
    else if (event == EVENT_OBJECT_NAMECHANGE) {
        // This event occurs when the window title (or object name) changes.
        // We only log this if it's the top-level window and if it's a browser.
        if (idObject == OBJID_WINDOW && hwnd == g_currentForegroundWindow) {
            if (IsBrowserWindow(hwnd)) {
                PrintTimestamp();
                LogWindowTitle(hwnd, "Window Focus Changed");
            }
        }
    }
}

int main()
{
    std::string exeDir = getExecutableDirectory();
    std::string filesDir = exeDir + "\\Files";
    if (!createDirectoryIfNotExists(filesDir)) {
        std::cerr << "Error: Could not create or access directory: " << filesDir << std::endl;
        return 1;
    }

    std::string csvFilePath = filesDir + "\\timestamps.csv";
    if (!fileExists(csvFilePath)) {
        std::cerr << "Error: " << csvFilePath 
                  << " does not exist. Please create it before running.\n";
        return 1;
    }

    // Directory and file exist, open the file.
    outputFile.open(csvFilePath);
    if (!outputFile.is_open()) {
        std::cerr << "Could not open file for writing: " << csvFilePath << std::endl;
        return 1;
    }

    std::thread t;

    // Set hook for when a window gains foreground (EVENT_SYSTEM_FOREGROUND)
    HWINEVENTHOOK hForegroundHook = SetWinEventHook(
        EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_FOREGROUND,
        NULL, WinEventProc, 0, 0, WINEVENT_OUTOFCONTEXT
    );

    // Set hook for when a window (or other UI object) changes its name/title (EVENT_OBJECT_NAMECHANGE)
    HWINEVENTHOOK hNameChangeHook = SetWinEventHook(
        EVENT_OBJECT_NAMECHANGE, EVENT_OBJECT_NAMECHANGE,
        NULL, WinEventProc, 0, 0, WINEVENT_OUTOFCONTEXT
    );

    // Error handling
    if (!hForegroundHook || !hNameChangeHook) {
        printf("Error setting up event hooks\n");
        return 1;
    }

    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    t.join();

    // Unhook events on exit
    UnhookWinEvent(hForegroundHook);
    UnhookWinEvent(hNameChangeHook);

    outputFile.close();

    return 0;
}
