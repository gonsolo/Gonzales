struct CameraSample {

        init(film: TwoRandomVariables = (0, 0), lens: TwoRandomVariables = (0, 0)) {
                self.film = film
                self.lens = lens
        }

        let film: TwoRandomVariables
        let lens: TwoRandomVariables
}
