#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

int main(void) {
    int server = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in address = {
        .sin_family = AF_INET,
        .sin_port = htons(8765),
        .sin_addr.s_addr = htonl(INADDR_ANY)
    };
    if (server < 0 || bind(server, (struct sockaddr *)&address, sizeof(address)) ||
        listen(server, 8)) {
        return 1;
    }
    int client = accept(server, 0, 0);
    char request[2048];
    const char response[] =
        "HTTP/1.1 200 OK\r\nContent-Length: 9\r\nConnection: close\r\n\r\nengine-ok";
    if (client < 0 || read(client, request, sizeof(request)) < 1 ||
        write(client, response, sizeof(response) - 1) != sizeof(response) - 1) {
        return 2;
    }
    close(client);
    close(server);
    return 0;
}
