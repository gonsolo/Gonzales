struct MediumInterface: Sendable {

        init(interior: String, exterior: String) {
                self.interior = interior
                self.exterior = exterior
        }

        let interior: String
        let exterior: String
}
